/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.funlbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.ProbabilisticTools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.ToolsRDD.joinLBMRowPartitionToData
import breeze.linalg.{DenseMatrix, DenseVector, min}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.Random

object Initialization  {

  def initialize(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                 condLatentBlock: FunLatentBlock,
                 n:Int, p :Int,
                 fullCovariance: Boolean,
                 maxPcaAxis: Int,
                 verbose:Boolean = true,
                 initMethod: String = "randomPartition")(implicit ss: SparkSession): (FunLatentBlockModel, List[Int]) = {

    val K = condLatentBlock.getK
    val L = condLatentBlock.getL

    initMethod match {
      case "randomPartition" => initFromRandomPartition(periodogram,K,L,n,p,fullCovariance, maxPcaAxis, verbose)
      case _ => {
        println(s"Warning: No initial method has been provided and initMethod $initMethod provided " +
          "does not match possible initialization method name (\"random\",\"sample\")" +
          "Continuing with random initialization..")
        initFromRandomPartition(periodogram,K,L,n,p,fullCovariance, maxPcaAxis, verbose)
      }
    }
  }

  def initFromRandomPartition(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                              K:Int,
                              L:Int,
                              n: Int,
                              p:Int,
                              fullCovariance: Boolean,
                              maxPcaAxis: Int,
                              verbose: Boolean = true)(implicit ss: SparkSession): (FunLatentBlockModel, List[Int]) = {


    if(verbose) println("Random Partition Initialization begins")
    val rowPartition: List[Int] = Random.shuffle((0 until n).map(i => i%K)).toList
    val colPartition: List[Int] = Random.shuffle((0 until p).map(j => j%L)).toList
    if(verbose) println("shuffling done")
    val periodogramWithRowPartition = joinLBMRowPartitionToData(periodogram, rowPartition,n)
    if(verbose) println("Init Maximization Step begins")
    val initModel = new FunLatentBlockModel(K,L).SEMGibbsMaximizationStep(
      periodogramWithRowPartition,
      colPartition,
      fullCovariance,
      true,
      maxPcaAxis,
      verbose)

    (initModel,colPartition)

  }


//  def initFromFunCLBMOnSample(dataCells: RDD[(Int, Array[DenseVector[Double]])],
//                                KVec: List[Int],
//                                proportionSample:Double = 0.2,
//                                nTry: Int = 5,
//                                nConcurrentPerTry: Int = 3)(implicit ss: SparkSession): (CondLatentBlockModel, List[Int]) = {
//
//    require(proportionSample>0 & proportionSample<=1, "proportionSample argument is a proportion (should be >0 and <=1)")
//
//    val resList = (0 until nTry).map(i => {
//      val dataSample = dataCells.sample(withReplacement = false, proportionSample)
//      val CLBM = new FunCondLatentBlock().setKVec(KVec)
//      CLBM.run(dataSample, nConcurrent=nConcurrentPerTry, nTryMaxPerConcurrent=10,initMethod = "random")
//    })
//
//    val allLikelihoods: DenseVector[Double] = DenseVector(resList.map(_("LogLikelihood").asInstanceOf[List[Double]].last).toArray)
//    val bestRes = resList(argmax(allLikelihoods))
//
//    (bestRes("Model").asInstanceOf[CondLatentBlockModel], bestRes("ColPartition").asInstanceOf[List[Int]])
//  }

}

