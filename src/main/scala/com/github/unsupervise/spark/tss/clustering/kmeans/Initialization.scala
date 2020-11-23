/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.kmeans

import com.github.unsupervise.spark.tss.clustering.lbmtools.ProbabilisticTools._
import breeze.linalg.{*, DenseMatrix, DenseVector, min, sum}
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec

object Initialization  {

  def initialize(data: RDD[(Int, Array[DenseVector[Double]])],
                 KMeans: KMeans,
                 n:Int,
                 verbose:Boolean = true,
                 initMethod: String = "random"): KMeansModel = {

    val K = KMeans.getK
    val nSample = min(n, 20)
    initMethod match {
      case "random" => Initialization.randomModelInitialization(data, K, nSample)
      case "KMeansPP" => {
        Initialization.KMeansPPInitialization(data, K, n, Tools.dist)
      }
      case _ => {
        println(s"Warning: No initial method has been provided and initMethod $initMethod provided " +
          "does not match possible initialization method name (\"random\",\"KMeansPP\") " +
          "Continuing with random initialization..")
        Initialization.randomModelInitialization(data, K, nSample)
      }
    }
  }

  def randomModelInitialization(data: RDD[(Int, Array[DenseVector[Double]])],
                                K: Int,
                                nSamples:Int = 10): KMeansModel = {


    val initialCenters: List[Array[DenseVector[Double]]] = (0 until K).map(k => {
      val sampleBlock: Array[Array[DenseVector[Double]]] = data.takeSample(false, nSamples).map(_._2)
      val newCenter: Array[DenseVector[Double]] = sampleBlock.reduce(Tools.sumArrayDV).map(e => e/nSamples.toDouble)
      newCenter
    }).toList

    KMeansModel(initialCenters)
  }

  def KMeansPPInitialization(data: RDD[(Int, Array[DenseVector[Double]])],
                             K: Int,
                             n: Int,
                             distance: (Array[DenseVector[Double]], Array[DenseVector[Double]]) => Double): KMeansModel = {

    val initialCenter = data.take(1).head._2
    @tailrec
    def go(currentCenters: List[Array[DenseVector[Double]]]): List[Array[DenseVector[Double]]] = {

      val distanceToCenters: DenseMatrix[Double] = DenseMatrix(currentCenters.map(center => {
        Tools.distanceToCenter(data, center, distance)
      }): _*).reshape(n, currentCenters.length)

      val minDistanceToCenters: DenseVector[Double] = min(distanceToCenters(*, ::))
      val minDistanceToCentersNormalized: DenseVector[Double] = minDistanceToCenters / sum(minDistanceToCenters)
      val idx = sample(minDistanceToCentersNormalized.toArray.toList)
      val newCenters:List[Array[DenseVector[Double]]] = currentCenters ++ List(data.filter(_._1==idx).collect().head._2)

      if (newCenters.length == K) {
        newCenters
      } else {
        go(newCenters)
      }
    }

    val centers = if (K > 1) {
      go(List(initialCenter))
    } else {
      List(initialCenter)
    }

    KMeansModel(centers)
  }

}

