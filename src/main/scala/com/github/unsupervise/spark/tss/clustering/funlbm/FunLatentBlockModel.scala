/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.funlbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.ToolsRDD._
import Tools._
import com.github.unsupervise.spark.tss.clustering.lbm.Tools._
import com.github.unsupervise.spark.tss.clustering.lbm.LatentBlockModel
import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.numerics.log
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

class FunLatentBlockModel(proportionsRows: List[Double],
                          proportionsCols: List[Double],
                          gaussians: List[List[MultivariateGaussian]],
                          var loadings: List[List[DenseMatrix[Double]]],
                          var centerings: List[List[DenseVector[Double]]])
  extends LatentBlockModel (proportionsRows, proportionsCols, gaussians){

  // Auxiliary constructor also takes a String?? (compile error)
  def this() {
    this(
      List(0D),
      List(0D),
      List(List(new MultivariateGaussian(
        Vectors.dense(Array(0D)),
        denseMatrixToMatrix(DenseMatrix(1D))))),
      List(List(DenseMatrix(0D))),
      List(List(DenseVector(0D))))
  }

  // Auxiliary constructor
  def this(K:Int,
           L:Int) {
    this(
      List.fill(K)(1/K.toDouble),
      List.fill(L)(1/L.toDouble),
      List.fill(L)(List.fill(K){new MultivariateGaussian(
        Vectors.dense(Array(0D)),
        denseMatrixToMatrix(DenseMatrix(1D)))}),
      List.fill(L)(List.fill(K){DenseMatrix(0D)}),
      List.fill(L)(List.fill(K){DenseVector(0D)}))
  }

  override def computeJointLogDistribRowsFromSample(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                                                    colPartition: List[Int])(implicit ss: SparkSession):
  RDD[(Int, Array[DenseVector[Double]], List[Double])] = {

    val logPiRows: DenseVector[Double] = DenseVector(this.proportionsRows.map(log(_)).toArray)
    data.map(row => {
      (row._1,
        row._2,
        (0 until this.K).map(k => {
          row._2.indices.map(j => {
            this.gaussians(colPartition(j))(k)
              .logpdf(Vectors.dense(
                (loadings(colPartition(j))(k)*
                  (row._2(j)-centerings(colPartition(j))(k))
                  ).toArray))
          }).sum + logPiRows(k)
        }).toList)
    }).cache()
  }

  override def computeJointLogDistribColsFromSample(data: RDD[(Int, Array[DenseVector[Double]], Int)])
                                                   (implicit ss: SparkSession): List[List[Double]] = {

    val logPiCols: DenseVector[Double] = DenseVector(this.proportionsCols.map(log(_)).toArray)
    val D: RDD[DenseMatrix[Double]] =
      data.map(row => {
        row._2.indices.map(j => {
          DenseMatrix((0 until this.L).map(l => {
            this.gaussians(l)(row._3)
              .logpdf(Vectors.dense((loadings(l)(row._3)*
                (row._2(j)-centerings(l)(row._3))).toArray))
          }).toArray)
        }).reduce((a, b) => DenseMatrix.vertcat(a, b))
      }).cache()

    val prob = D.reduce(_ + _)
    val sumProb = prob(*, ::).map(dv => dv.toArray.toList).toArray.toList.zipWithIndex.map(e =>
      (DenseVector(e._1.toArray) + logPiCols).toArray.toList)

    sumProb
  }

  def SEMGibbsMaximizationStep(periodogram: RDD[(Int, Array[DenseVector[Double]], Int)],
                               colPartition: List[Int],
                               fullCovariance: Boolean,
                               updateLoadings: Boolean,
                               maxPcaAxis: Int,
                               verbose: Boolean)(implicit ss: SparkSession): FunLatentBlockModel = {
    periodogram.cache()

    println(colPartition)
    require(colPartition.distinct.length == L)

    val partitionPerColBc: Broadcast[DenseVector[Int]] = ss.sparkContext.broadcast(DenseVector(colPartition: _*))
    val (newCenterings, newLoadings) = if (updateLoadings) {
      val (means, covMat, _, _) = getMeansAndCovariances(periodogram, partitionPerColBc, L, K, fullCovariance)
      val newLoadings =
        (0 until L).map(l => {
          (0 until K).map(k => {
            getLoadings(covMat(l)(k), maxPcaAxis)
          }).toList
        }).toList
      (means, newLoadings)
    } else {
      (this.centerings, this.loadings)
    }

    val dataLowerSubSpace = periodogram.map(r =>
      (r._1,
        r._2.indices.map(j => {
          newLoadings(colPartition(j))(r._3) * (r._2(j) - newCenterings(colPartition(j))(r._3))
        }
        ).toArray,
        r._3))

    val (means, covMat, sizeBlock, _) = getMeansAndCovariances(dataLowerSubSpace, partitionPerColBc, L, K, fullCovariance)
    val newModels: List[List[MultivariateGaussian]] =
      (0 until L).map(l => {
        (0 until K).map(k => {
          new MultivariateGaussian(
            Vectors.dense(means(l)(k).toArray),
            denseMatrixToMatrix(covMat(l)(k)))
        }).toList
      }).toList

    val proportions = proportionFromLBMSizeBlock(sizeBlock)

    new FunLatentBlockModel(proportions._1, proportions._2, newModels, newLoadings, newCenterings)
  }

}

