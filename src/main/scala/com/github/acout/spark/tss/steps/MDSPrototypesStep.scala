package com.github.unsupervise.spark.tss.steps

import java.io.{BufferedWriter, File, FileWriter}

import com.github.unsupervise.spark.tss.core.{PrototypeDistsF, TSS}

/**
  * Lazy Multi-Dimensional Scaling (MDS) Dimension Reduction from pairwise clustering prototype distance matrix Step Command.
  * This step first precomputes (univariate) clusters prototype to prototype distances matrix, cache it,
  * and use it to compute (multivariate) distances between series depending on their per univariate member
  * hard membership to clusters.
  * The class is designed to perform several MDS computations at once, thus works with Array parameters.
  * @param outColNames the output column names containing the reduced dimensions obtained by each MDS computation
  * @param outWeightedColNames the output column names containing the weighted reduced dimensions obtained by each MDS computation
  * @param inColNames the input column names containing the original representations from which performing each MDS computation
  * @param prototypesFilePaths the clustering prototype / centers file paths to use for distance computations for each MDS computation
  * @param maxDimensionsNumber the maximum number of dimensions to allow for the output reduced representations
  * @param explainedVarianceThreshold the threshold value to automatically select the number of dimensions in the output reduced representations (if not reached, <maxDimensionsNumber> is taken instead)
  * @param partitionsNumber the number of partitions to enforce during the computation
  * @param runsNb the number of multiple runs to perform for each MDS computation
  * @param eigenValuesOutFilePaths the output file paths where to store each MDS eigenvalues set
  * @param varianceProportionsOutFilePaths the output file paths where to store each MDS variance proportions of reduced coordinates
  * @param stressOutFilePaths the output file paths where to store each MDS stress value
  */
case class MDSPrototypesStep(outColNames: Array[String], outWeightedColNames: Array[String], inColNames: Array[String], prototypesFilePaths: Array[String],
                             maxDimensionsNumber: Int, explainedVarianceThreshold: Double, partitionsNumber: Int, runsNb: Int, eigenValuesOutFilePaths: Array[String],
                             varianceProportionsOutFilePaths: Array[String], stressOutFilePaths: Array[String]) extends Step {

  override def apply(in: TSS): TSS = {

    implicit val ss = in.series.sparkSession

    outColNames.indices.foldLeft(in){
      case (tss, i) => {
        import tss.series.sparkSession.implicits._
        //TODO: Cleaner?
        val clusteringPrototypes = tss.series.sparkSession.sparkContext.textFile(prototypesFilePaths(i).replace("/dbfs", "dbfs:"))
          .map(r => {
            val splittedRow = r.split(",")
            val pk = splittedRow(0).toLong
            (pk, splittedRow.drop(1).map(_.toDouble))
          }).toDF.as[(Long, Seq[Double])]//.persist
        //clusteringPrototypes.show(30)
        val clusteringPrototypeDistances = TSS.distances(clusteringPrototypes, TSS.euclideanF[Seq[Double]](false)).getAs2DArray
        val protoDistancesFile = new File(eigenValuesOutFilePaths(i).replace(".csv", ".protoDistances.csv"))
        val writer = new BufferedWriter(new FileWriter(protoDistancesFile))
        writer.write(clusteringPrototypeDistances.map(_.mkString(",")).mkString("\n"))
        writer.close()
        //println("Proto distances: " + clusteringPrototypeDistances.map(_.mkString(",")).mkString("\n"))
        val broadcastedDistances = tss.series.sparkSession.sparkContext.broadcast(clusteringPrototypeDistances)
        val prototypeDistancesHandler = PrototypeDistsF.sum(broadcastedDistances)
        //try{
        println(tss.select("scenarioId").series.take(100).map(_.getString(0)).mkString(","))
        tss.persist
        val tss2 =
          tss.addMDS(/*tss, */inColNames(i) + "_MDS", inColNames(i), maxDimensionsNumber, prototypeDistancesHandler.apply _, explainedVarianceThreshold, partitionsNumber, runsNb, 0D,
            Some(new File(eigenValuesOutFilePaths(i))), Some(new File(varianceProportionsOutFilePaths(i))), Some(new File(stressOutFilePaths(i))))
            /*val tss2 = addMDS(tss, inColNames(i) + "_MDS", inColNames(i), maxDimensionsNumber, prototypeDistancesHandler.apply _, explainedVarianceThreshold, partitionsNumber, runsNb,
                  Some(new File(eigenValuesOutFilePaths(i))), Some(new File(varianceProportionsOutFilePaths(i))), Some(new File(stressOutFilePaths(i))))*/
            //Scale MDS results to enforce the correct relative importance of each feature to the afterwards weighting
            .addMLVectorized(inColNames(i) + "_MDSVec", inColNames(i) + "_MDS")
            .addColScaled(outColNames(i), inColNames(i) + "_MDSVec", true, true)
            .addSeqFromMLVector(outColNames(i) + "Seq", outColNames(i))
            .coalesce(partitionsNumber)
        /*
        val reducedVarContributions = ss.sparkContext.textFile(dbfsMainFolderToAnalysePath + "/" + cl + "MDS.varianceProportions.csv").map(_.split(",").map(_.toDouble)).first
    val reducedVarContributions = ss.sparkContext.textFile(dbfsMainFolderToAnalysePath + "/PCAVariances.csv").map(_.split(",")(1).toDouble/* / 100D*/).collect.map(x => scala.math.sqrt(x))
         */
        Thread.sleep(2000) //Avoid to load partial file
        val reducedVarContributions = tss.series.sparkSession.sparkContext.textFile(varianceProportionsOutFilePaths(i).replace("/dbfs", "dbfs:")).map(_.split(",").map(_.toDouble)).first
        println("Contributions: " + reducedVarContributions.mkString(","))
        println("Contributions Sqrt: " + reducedVarContributions.map(x => scala.math.sqrt(x)).mkString(","))
        val tss3 = tss2.addWeighted(inColNames(i) + "_MDSWeightedColScaled", outColNames(i) + "Seq", reducedVarContributions.map(x => scala.math.sqrt(x)))
          .addWeighted(inColNames(i) + "_MDSWeighted", inColNames(i) + "_MDS", reducedVarContributions.map(x => scala.math.sqrt(x)))
          .addMLVectorized(outWeightedColNames(i), inColNames(i) + "_MDSWeighted")
        //Remove Intermediate Columns
        //tss3.drop(List("_MDS", "_MDSVec", "_MDSWeighted", "_MDSVecWeighted").map(x => inColNames(i) + x): _*)
        tss3
      }
    }
  }
}
