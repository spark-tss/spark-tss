/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.funclbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.ProbabilisticTools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.ToolsRDD._
import com.github.unsupervise.spark.tss.core.TSS
import com.github.unsupervise.spark.tss.clustering.kmeans.KMeans
import com.github.unsupervise.spark.tss.clustering.funlbm.FunLatentBlock
import breeze.linalg.{DenseMatrix, DenseVector, max, min, diag}
import org.apache.spark.sql.functions.{col}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.Random

object Initialization  {

  def initialize(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                 condLatentBlock: FunCondLatentBlock,
                 n:Int, p :Int,
                 fullCovariance: Boolean,
                 maxPcaAxis: Int,
                 verbose:Boolean = true,
                 initMethod: String = "randomPartition")(implicit ss: SparkSession): (FunCondLatentBlockModel, List[Int]) = {

    val KVec = condLatentBlock.KVec
    initMethod match {
//      case "random" => {
//        val model = Initialization.initFromComponentSample(periodogram, KVec, nSampleForLBMInit,verbose)
//        (model, (0 until p).map(j => sample(model.proportionsCols)).toList)
//      }
      case "randomPartition" => {
        initFromRandomPartition(periodogram, KVec, n,p,fullCovariance,maxPcaAxis,verbose)
      }
      case "KMeans" => {
        Initialization.initFromColKMeans(periodogram,KVec,n,fullCovariance, maxPcaAxis, verbose)
      }
      case "FunLBM" => {
        Initialization.initFromFunLBM(periodogram,KVec,n,fullCovariance,maxPcaAxis,verbose)
      }
      case _ => {
        println(s"Warning: No initial method has been provided and initMethod $initMethod provided " +
          "does not match possible initialization method name (\"random\",\"randomPartition\",\"KMeans\",\"FunLBM\")" +
          "Continuing with random partition initialization..")
        initFromRandomPartition(periodogram, KVec, n,p,fullCovariance, maxPcaAxis, verbose)
      }
    }
  }


//  def initFromComponentSample(periodogram: RDD[(Int, Array[DenseVector[Double]])],
//                              KVec: List[Int],
//                              nSamples:Int = 10,
//                              verbose: Boolean=false)(implicit ss: SparkSession): FunCondLatentBlockModel = {
//
//    if(verbose) println("Random Sample Initialization")
//
//    // %%%%%%%%%%%%%%%%%%%%%%
//    val flattenedPeriodogram: RDD[DenseVector[Double]] = periodogram.map(_._2).flatMap(row => row.toArray.toList)
//    val flattenedRDDAsList : RDD[(Int, Int, List[Double])] = flattenedPeriodogram.map(r => (0,0, r.toArray.toList))
//    val (unsortedPcaCoefs, loadings) = getPcaAndLoadings(flattenedRDDAsList)
//    val pcaCoefs: RDD[(Int, Array[DenseVector[Double]])] = periodogram.map(row => (row._1, row._2.map(e => loadings * e)))
//    // %%%%%%%%%%%%%%%%%%%%%%
//
//    val L = KVec.length
//    val MultivariateGaussians: List[List[MultivariateGaussian]] = KVec.indices.map(l => {
//      (0 until KVec(l)).map(k_l => {
//        val sampleBlock: List[DenseVector[Double]] = pcaCoefs.takeSample(withReplacement = false, nSamples)
//          .map(e => Random.shuffle(e._2.toList).head).toList
//        val mode: DenseVector[Double] = meanListDV(sampleBlock)
//        new MultivariateGaussian(Vectors.dense(mode.toArray),denseMatrixToMatrix(covariance(sampleBlock, mode)))
//      }).toList
//    }).toList
//
//    val rowProportions:List[List[Double]] = (0 until L).map(l => {List.fill(KVec(l))(1.0 / KVec(l))}).toList
//    val colProportions:List[Double] =  List.fill(L)(1.0 / L):List[Double]
//    val loadingsList = (0 until L).map(l => {List.fill(KVec(l))(loadings)}).toList
//    FunCLBMSpark.FunCondLatentBlockModel(rowProportions, colProportions, loadingsList, MultivariateGaussians)
//  }


  def initFromRandomPartition(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                              KVec: List[Int],
                              n: Int,
                              p: Int,
                              fullCovariance: Boolean,
                              maxPcaAxis: Int,
                              verbose: Boolean=false)(implicit ss: SparkSession): (FunCondLatentBlockModel, List[Int]) = {
    if(verbose) println("Random Partition Initialization")
    val colPartition: List[Int] = Random.shuffle((0 until p).map(j => j%KVec.length)).toList
    val randomRowPartition: List[List[Int]] = KVec.indices.map(l => {
      Random.shuffle((0 until n).map(_%KVec(l))).toList
    }).toList

    val periodogramWithRowPartition = joinRowPartitionToData(periodogram, randomRowPartition,n)
    val initModel = new FunCondLatentBlockModel(KVec).SEMGibbsMaximizationStep(
      periodogramWithRowPartition,
      colPartition,
      fullCovariance,
       true,
      maxPcaAxis,
      verbose)

    (initModel,colPartition)
  }

  def initFromColKMeans(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                        KVec: List[Int],
                        n: Int,
                        fullCovariance: Boolean,
                        maxPcaAxis: Int,
                        verbose:Boolean=false)(implicit ss: SparkSession): (FunCondLatentBlockModel, List[Int]) = {


    if(verbose) println("KMeans Initialization")

    // %%%%%%%%%%%%%%%%%%%%%%
    val flattenedPeriodogram: RDD[DenseVector[Double]] = periodogram.map(_._2).flatMap(row => row.toArray.toList)
    val flattenedRDDAsList : RDD[(Int, Int, List[Double])] = flattenedPeriodogram.map(r => (0,0, r.toArray.toList))
    //TODO: Find another way than getPcaAndLoadings ??
    val (unsortedPcaCoefs, loadings) = getPcaAndLoadings(flattenedRDDAsList)
    val pcaCoefs: RDD[(Int, Array[DenseVector[Double]])] = periodogram.map(row => (row._1, row._2.map(e => loadings * e)))
    // %%%%%%%%%%%%%%%%%%%%%%

    val L = KVec.length
    val p = pcaCoefs.take(1).head._2.length

    val dataByCol = inverseIndexedList(pcaCoefs.collect().toList)
    val dataByColRDD = ss.sparkContext.parallelize(dataByCol,200)
    val kmeanCol = new KMeans()
    val resModelCol = kmeanCol.setK(L).run(dataByColRDD, verbose=verbose)
    val colPartition = resModelCol("Partition").asInstanceOf[List[Int]]

    println(colPartition)
    val rowPartition = (0 until L).map(l => {
      val filteredDataByCol = dataByCol.filter(r => colPartition(r._1)==l)
      val filteredDataByRow = inverseIndexedList(filteredDataByCol)
      val filteredDataByRowRDD = ss.sparkContext.parallelize(filteredDataByRow,100)
      val kmeanRow = new KMeans().setK(KVec(l))
      val resModelRow = kmeanRow.run(filteredDataByRowRDD, verbose=verbose)
      println(resModelRow("Partition").asInstanceOf[List[Int]])
      resModelRow("Partition").asInstanceOf[List[Int]]
    }).toList

    val periodogramWithRowPartition = joinRowPartitionToData(periodogram, rowPartition, n)

    val initModel =new FunCondLatentBlockModel(KVec).SEMGibbsMaximizationStep(
      periodogramWithRowPartition,
      colPartition,
      fullCovariance,
      true,
      maxPcaAxis,
      verbose)

    (initModel,colPartition)
  }

  def initFromFunLBM(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                      KVec: List[Int],
                      n:Int,
                      fullCovariance: Boolean,
                      maxPcaAxis: Int,
                      verbose:Boolean = false)(implicit ss: SparkSession): (FunCondLatentBlockModel, List[Int]) = {
    if(verbose) println("FunLBM Initialization")

    val L = KVec.length
    val maxK = max(KVec)
    val dataByCol = inverseIndexedList(periodogram.collect().toList)

    val latentBlock = new FunLatentBlock()
    latentBlock.setMaxIterations(5).setMaxBurninIterations(5).setL(L).setK(maxK).setUpdateLoadingStrategy(4)
    val resLBM = latentBlock.run(periodogram, verbose= true,
    nConcurrent = 1, nTryMaxPerConcurrent = 10)
    val colPartition: List[Int] = resLBM("ColPartition").asInstanceOf[List[Int]]

    val rowPartition = (0 until L).map(l => {
      val filteredDataByCol = dataByCol.filter(r => colPartition(r._1)==l)
      val filteredDataByRow = inverseIndexedList(filteredDataByCol)
      val filteredDataByRowRDD = ss.sparkContext.parallelize(filteredDataByRow,100)
      val latentBlock = new FunCondLatentBlock().setKVec(List(KVec(l))).setUpdateLoadingStrategy(4)
      val resModelRow = latentBlock.run(filteredDataByRowRDD, verbose=true, initMethod ="random",
        nConcurrent = 1, nTryMaxPerConcurrent = 10)
      val rowPartition = resModelRow("RowPartition").asInstanceOf[List[List[Int]]].head
      rowPartition
    }).toList
    val periodogramWithRowPartition = joinRowPartitionToData(periodogram, rowPartition, n)
    val initModel =new FunCondLatentBlockModel(KVec).SEMGibbsMaximizationStep(
      periodogramWithRowPartition,
      colPartition,
      fullCovariance,
      true,
      maxPcaAxis,
      verbose)
    (initModel,colPartition)
  }

  //TODO: Remove to use getMeansAndCovariances ?? For Consistency
  def getPcaAndLoadings(dataRDD: RDD[(Int, Int, List[Double])], maxK:Int = 20)(implicit ss: SparkSession):
  (RDD[(Int, Int, DenseVector[Double])], DenseMatrix[Double]) = {

    val dfWithSchema = ss.createDataFrame(dataRDD).toDF("scenario_id", "varName", "periodogram")

    val tss = new TSS(dfWithSchema, forceIds = false)

    val tssVec = tss.addMLVectorized("periodogramMLVec", "periodogram")

    val (joinedTSS, pcaRes) = tssVec.addPCA_("logInterpolatedDFTPeriodogram_PCAVec", "periodogramMLVec", maxK, 0.99)
    val eigenValue = diag(DenseVector(pcaRes.explainedVariance.toArray))
    val pcDM = matrixToDenseMatrix(pcaRes.pc)
    val loadings = denseMatrixToMatrix(pcDM*eigenValue).toDense

    //Scale PCA results to enforce the correct relative importance of each feature to the afterwards weighting
    val scaledJoinedTSS = joinedTSS.addColScaled("logInterpolatedDFTPeriodogram_ColScaledPCAVec",
      "logInterpolatedDFTPeriodogram_PCAVec", scale = true, center = true)
      .addSeqFromMLVector("pcaCoordinatesV", "logInterpolatedDFTPeriodogram_ColScaledPCAVec")
      //Drop intermediate columns for cleaner output
      .drop("logInterpolatedDFTPeriodogram_ColScaledPCAVec", "logInterpolatedDFTPeriodogram_PCAVec", "periodogram")

    val series = scaledJoinedTSS.select("scenario_id","varName","pcaCoordinatesV").series
    val pcaCoefs = series.select(col("scenario_id"),
          col("varName"),
          col("pcaCoordinatesV")).rdd

    val pcaCoefsRDDs = pcaCoefs.map(row => (
      row.getInt(0),
      row.getInt(1),
      DenseVector[Double](row.getSeq[Double](2).toArray))
    )

    (pcaCoefsRDDs, matrixToDenseMatrix(loadings).t)

  }
}

