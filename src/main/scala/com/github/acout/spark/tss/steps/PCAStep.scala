package com.github.unsupervise.spark.tss.steps

import java.io.File

import com.github.unsupervise.spark.tss.core.TSS

/**
  * Lazy PCA Dimension Reduction Step Command.
  * The class is designed to perform several PCA computations at once, thus works with Array parameters.
  * @param outColNames the output column names containing the reduced coordinates of each PCA computation.
  * @param outWeightedColNames the output column names containing the weighted reduced coordinates of each PCA computation.
  * @param inColNames the input column names containing the input representations to use for each PCA computation.
  * @param maxDimensionsNumber the maximum number of reduced coordinates allowed for each PCA output.
  * @param explainedVarianceThreshold the threshold allowing to choose the number of reduced coordinates for each PCA output. If the threshold is not reached with less than <maxDimensionsNumber> reduced coordinates, <maxDimensionsNumber> ooordinates are taken instead.
  * @param pcaLoadingsOutFilePaths the Array of file paths where to store each PCA loading matrix
  * @param pcaVariancesOutFilePaths the Array of file paths where to store each PCA variance contributions of reduced coordinates
  * @param scaleBeforePCA whether to scale by column the input series before computing PCA
  */
case class PCAStep(outColNames: Array[String], outWeightedColNames: Array[String], inColNames: Array[String], maxDimensionsNumber: Int, explainedVarianceThreshold: Double,
                   pcaLoadingsOutFilePaths: Array[String], pcaVariancesOutFilePaths: Array[String], scaleBeforePCA: Boolean) extends Step {

  def this(outColName: String, outWeightedColName: String, inColName: String, maxDimensionsNumber: Int, explainedVarianceThreshold: Double, pcaLoadingsOutFilePath: String, pcaVariancesOutFilePath: String, scaleBeforePCA: Boolean) =
    this(Array(outColName), Array(outWeightedColName), Array(inColName), maxDimensionsNumber, explainedVarianceThreshold, Array(pcaLoadingsOutFilePath), Array(pcaVariancesOutFilePath), scaleBeforePCA)

  override def apply(in: TSS): TSS = {

    outColNames.indices.foldLeft(in){
      case (tss, i) => {

        val tss2 = {
          if(scaleBeforePCA) tss.addColScaled(inColNames(i) + "_ScaledVecColScaled", inColNames(i), true, true) else tss
        }

        val joinedTSS =
          tss2.addPCA(inColNames(i) + "_PCAVec", if(scaleBeforePCA) inColNames(i) + "_ScaledVecColScaled" else inColNames(i), maxDimensionsNumber, explainedVarianceThreshold,
            Some(new File(pcaLoadingsOutFilePaths(i))), Some(new File(pcaVariancesOutFilePaths(i))))
            //Scale PCA results to enforce the correct relative importance of each feature to the afterwards weighting
            .addColScaled(outColNames(i), inColNames(i) + "_PCAVec", true, true)
            .addSeqFromMLVector(outColNames(i) + "_ColScaledPCASeq", outColNames(i))

        //pcaAugmentedTSS.addSeqFromMLVector("pcaCoordinatesW", "pcaCoordinatesVW")

        Thread.sleep(2000) //Avoid to load partial file

        val reducedVarContributions = tss.series.sparkSession.sparkContext.textFile(pcaVariancesOutFilePaths(i).replace("/dbfs", "dbfs:")).map(_.split(",")(1).toDouble).collect.map(x => scala.math.sqrt(x))
        //println("Contributions: " + reducedVarContributions.mkString(","))
        val joinedTSS2 = joinedTSS.addWeighted(outColNames(i) + "_ColScaledWeighted", outColNames(i) + "_ColScaledPCASeq", reducedVarContributions)
          .addMLVectorized(outWeightedColNames(i), outColNames(i) + "_ColScaledWeighted")

        //joinedTSS2.select(inColNames(i) + "_ScaledVecColScaledPCAVec", outColNames(i)).series.show(7)

        //Remove Intermediate columns
        joinedTSS2/*.drop(
          List("_ScaledVec", "_ScaledVecColScaled", "_ScaledVecColScaledPCAVec", "_ScaledVecColScaledPCA", "_ScaledVecColScaledPCAWeighted").map(x => inColName + x): _*
        )*/
      }

    }

  }

}

object PCAStep {

  def apply(outColName: String, outWeightedColName: String, inColName: String, maxDimensionsNumber: Int, explainedVarianceThreshold: Double, pcaLoadingsOutFilePath: String, pcaVariancesOutFilePath: String, scaleBeforePCA: Boolean): PCAStep =
    PCAStep(Array(outColName), Array(outWeightedColName), Array(inColName), maxDimensionsNumber, explainedVarianceThreshold, Array(pcaLoadingsOutFilePath), Array(pcaVariancesOutFilePath), scaleBeforePCA)

}
