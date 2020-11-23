/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.funclbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.ProbabilisticTools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.ToolsRDD._
import com.github.unsupervise.spark.tss.clustering.funlbm.Tools._
import com.github.unsupervise.spark.tss.clustering.clbm.Tools._
import com.github.unsupervise.spark.tss.clustering.clbm.CondLatentBlockModel
import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.numerics.{exp, log}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


class FunCondLatentBlockModel(proportionsRows: List[List[Double]],
                              proportionsCols: List[Double],
                              gaussians: List[List[MultivariateGaussian]],
                              var loadings: List[List[DenseMatrix[Double]]],
                              var centerings: List[List[DenseVector[Double]]])

  extends CondLatentBlockModel (proportionsRows, proportionsCols, gaussians) {

  def this() {
    this(List(List(0D)),
      List(0D),
      List(List(new MultivariateGaussian(
        Vectors.dense(Array(0D)),
        denseMatrixToMatrix(DenseMatrix(1D))))),
      List(List(DenseMatrix(0D))),
      List(List(DenseVector(0D))))
  }


  def this(KVec: List[Int]) {
    this(
      KVec.indices.map(l => List.fill(KVec(l))(1 / KVec(l).toDouble)).toList,
      List.fill(KVec.length)(1 / KVec.length.toDouble),
      KVec.indices.map(l => List.fill(KVec(l))(new MultivariateGaussian(
        Vectors.dense(Array(0D)),
        denseMatrixToMatrix(DenseMatrix(1D))))).toList,
      KVec.indices.map(l => (0 until KVec(l)).map(_ => DenseMatrix(0D)).toList).toList,
      KVec.indices.map(l => (0 until KVec(l)).map(_ => DenseVector(0D)).toList).toList)
  }

  override def computeJointLogDistribRowsFromSample(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                                                    colPartition: List[Int]): RDD[(Int, Array[DenseVector[Double]], List[List[Double]])] = {

    val logCondPiRows: List[List[Double]] = this.proportionsRows.map(piRow=> piRow.map(log(_)))
    val rangeCols: List[List[Int]] = KVec.indices.map(l => colPartition.zipWithIndex.filter(_._1 == l).map(_._2)).toList

    data.map(row => {
      (row._1,
        row._2,
        rangeCols.indices.map(l => {
          (0 until KVec(l)).map(k_l => {
            rangeCols(l).map(col => {
              this.gaussians(l)(k_l).logpdf(Vectors.dense(
                (loadings(l)(k_l) * (row._2(col)-centerings(l)(k_l)))
                  .toArray))
            }).sum + logCondPiRows(l)(k_l)
          }).toList
        }).toList)
    })
  }

  override def computeJointLogDistribColsFromSample(data: RDD[(Int, Array[DenseVector[Double]], List[Int])]): List[List[Double]] = {

    val logPiCols: DenseVector[Double] = DenseVector(this.proportionsCols.map(log(_)).toArray)
    var t0 = System.nanoTime()
    val D1 = data.flatMap(row => {
      row._2.indices.map(j => {
        (j, DenseVector(KVec.indices.map(l => {
          this.gaussians(l)(row._3(l))
            .logpdf(Vectors.dense(
              (loadings(l)(row._3(l)) * (row._2(j)-centerings(l)(row._3(l))))
                .toArray))
        }).toArray))
      })
    })

    val D = D1.reduceByKey(_+_).collect().sortBy(_._1).map(_._2)
    val sumProb = D.map(e => (e+ logPiCols).toArray.toList).toList
    sumProb
  }

  def SEMGibbsMaximizationStep(periodogram: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                               colPartition: List[Int],
                               fullCovariance: Boolean,
                               updateLoadings: Boolean,
                               maxPcaAxis: Int,
                               verbose: Boolean)(implicit ss: SparkSession): FunCondLatentBlockModel = {
    periodogram.cache()
    require(colPartition.distinct.length == KVec.length)

    val partitionPerColBc: Broadcast[DenseVector[Int]] = ss.sparkContext.broadcast(DenseVector(colPartition: _*))
    val (meansPeriodogram, newLoadings) = if (updateLoadings) {
      if(verbose){println("Updating loadings")}
      val (means, covMat, _, _) = getMeansAndCovariances(periodogram, partitionPerColBc, KVec, true, verbose=verbose)
      val newLoadings =
        KVec.indices.map(l => {
          (0 until KVec(l)).map(k => {
            getLoadings(covMat(l)(k),maxPcaAxis)
          }).toList
        }).toList
      (means, newLoadings)
    } else {(this.centerings, this.loadings)}

    if(verbose){println("Projecting in lower subspace")}

    val dataLowerSubSpace = periodogram.map(r =>
      (r._1, r._2.indices.map(j => {
          newLoadings(colPartition(j))(r._3(colPartition(j))) * (r._2(j) - meansPeriodogram(colPartition(j))(r._3(colPartition(j))))
        }).toArray, r._3))

    if(verbose){println("Parameters inference")}
    val (means, covMat, sizeBlock, _) = getMeansAndCovariances(dataLowerSubSpace, partitionPerColBc, KVec, fullCovariance, verbose)
    val newModels: List[List[MultivariateGaussian]] =
      KVec.indices.map(l => {
        (0 until KVec(l)).map(k => {
          new MultivariateGaussian(
            Vectors.dense(means(l)(k).toArray),
            denseMatrixToMatrix(covMat(l)(k)))
        }).toList
      }).toList

    val proportions = proportionFromCLBMSizeBlock(sizeBlock)

    new FunCondLatentBlockModel(proportions._1, proportions._2, newModels, newLoadings, meansPeriodogram)
  }

}


