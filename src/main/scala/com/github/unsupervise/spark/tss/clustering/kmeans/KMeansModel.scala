/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.kmeans

import breeze.linalg.{DenseVector, argmin}
import org.apache.spark.rdd.RDD

case class KMeansModel(centers: List[Array[DenseVector[Double]]]) {
  val precision = 1e-8
  val K = centers.length
  // Auxiliary constructor also takes a String?? (compile error)
  def this() {
    this(List(Array(DenseVector(0D))))
  }
  // Version to call inside the SEM algorithm
  def ClassificationExpectationStep(data: RDD[(Int, Array[DenseVector[Double]], Int)]):
  RDD[(Int, Array[DenseVector[Double]], Int)] = {
    data.map(row => {
      val dist: DenseVector[Double] = DenseVector(centers.map(center => {Tools.dist(row._2, center)}).toArray)
      (row._1,
        row._2,
        argmin(dist)
      )
    })
  }

  def MaximizationStep(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                       verbose: Boolean = true): KMeansModel = {

      val dataAndSizeByCluster = (0 until K).map(k => {
        val filteredData = data.filter(_._3 == k)
        val sizeCluster = filteredData.count().toInt
        require(sizeCluster > 1, "Algorithm could not converge: empty block")
        (filteredData, sizeCluster)
      })

      val newCenters: List[Array[DenseVector[Double]]] = (0 until this.K).map(k => {
        val filteredRDD = dataAndSizeByCluster(k)._1
        val sizeCluster = dataAndSizeByCluster(k)._2
        val newCenter: Array[DenseVector[Double]] = filteredRDD.map(row =>
          row._2).reduce(Tools.sumArrayDV).map(e => e/sizeCluster.toDouble)
        newCenter
      }).toList

    KMeansModel(newCenters)
  }

}

