/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.lbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.ProbabilisticTools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.ToolsRDD._
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.Random

object Initialization  {

  def randomModelInitialization(data: RDD[(Int, Array[DenseVector[Double]])],
                                K:Int,
                                L:Int,
                                nSamples:Int=10,
                                sc:SparkContext): LatentBlockModel = {


    val MultivariateGaussians: List[List[MultivariateGaussian]] =
    (0 until L).map(l => {
      (0 until K).map(k => {
        val sampleBlock: List[DenseVector[Double]] = data.takeSample(false, nSamples)
          .map(e => Random.shuffle(e._2.toList).head).toList

        val mode: DenseVector[Double] = meanListDV(sampleBlock)
        new MultivariateGaussian(Vectors.dense(mode.toArray), denseMatrixToMatrix(covariance(sampleBlock, mode)))
      }).toList
    }).toList

    LatentBlockModel(List.fill(K)(1.0 / K):List[Double], List.fill(L)(1.0 / L):List[Double],MultivariateGaussians)
  }

  def initFromRandomPartition(data: RDD[(Int, Array[DenseVector[Double]])],
                              K:Int,
                              L:Int,
                              n: Int,
                              p:Int,
                              fullCovariance: Boolean,
                              verbose: Boolean = true)(implicit ss: SparkSession): (LatentBlockModel, List[Int]) = {


    if(verbose) println("Random Partition Initialization")

    val rowPartition: List[Int] = Random.shuffle((0 until n).map(i => i%K)).toList
    val colPartition: List[Int] = Random.shuffle((0 until p).map(j => j%L)).toList
    val dataWithRowPartition = joinLBMRowPartitionToData(data, rowPartition,n)

    var initModel =new LatentBlockModel(K,L).SEMGibbsMaximizationStep(dataWithRowPartition, colPartition, fullCovariance)
    (initModel,colPartition)
  }

}

