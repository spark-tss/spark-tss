/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.clbm

import Tools.getMeansAndCovariances
import com.github.unsupervise.spark.tss.clustering.lbmtools.ProbabilisticTools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.ToolsRDD._
import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.{exp, log}
import org.apache.spark.ml.linalg.{Matrices, Vectors}
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

case class CondLatentBlockModel(proportionsRows: List[List[Double]],
                                proportionsCols: List[Double],
                                gaussians: List[List[MultivariateGaussian]]) {
  val precision = 1e-5
  def KVec: List[Int] = gaussians.map(_.length)

  def this() {
    this(List(List(0D)),List(0D),List(List(new MultivariateGaussian(
      Vectors.dense(Array(0D)),
      denseMatrixToMatrix(DenseMatrix(1D))))))
  }


  def this(KVec:List[Int]) {
    this(
      KVec.indices.map(l => List.fill(KVec(l))(1/KVec(l).toDouble)).toList,
      List.fill(KVec.length)(1/KVec.length.toDouble),
      KVec.indices.map(l => List.fill(KVec(l))(new MultivariateGaussian(
        Vectors.dense(Array(0D)),
        denseMatrixToMatrix(DenseMatrix(1D))))).toList)
  }

  // Version to call inside the SEM algorithm
  // Version to call inside the SEM algorithm
  def SEMGibbsExpectationStep(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                              colPartition: List[Int],
                              nIter: Int = 3,
                              verbose: Boolean = true): (RDD[(Int, Array[DenseVector[Double]], List[Int])], List[Int]) = {

    var newData: RDD[(Int, Array[DenseVector[Double]], List[Int])] = drawRowPartition(data, colPartition)
    var newColPartition: List[Int] = drawColPartition(newData)
    var k: Int = 1
    while (k < nIter) {
      newData = drawRowPartition(data, newColPartition)
      newColPartition = drawColPartition(newData)
      k += 1
    }
    (newData, newColPartition)
  }

  def drawRowPartition(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                       colPartition: List[Int]): RDD[(Int, Array[DenseVector[Double]], List[Int])] = {

    val jointLogDistribRows = computeJointLogDistribRowsFromSample(data, colPartition)

    jointLogDistribRows.map(x => {
      (x._1,
        x._2,
        x._3.indices.map(l => {
          val LSE = logSumExp(x._3(l))
          sample(x._3(l).map(e => exp(e - LSE)))
        }).toList)
    })
  }

  def computeJointLogDistribRowsFromSample(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                                           colPartition: List[Int]): RDD[(Int, Array[DenseVector[Double]], List[List[Double]])] = {

    val logCondPiRows: List[List[Double]] = this.proportionsRows.map(piRow=> piRow.map(log(_)))
    val rangeCols: List[List[Int]] = KVec.indices.map(l => colPartition.zipWithIndex.filter(_._1 == l).map(_._2)).toList

    data.map(row => {
      (row._1,
        row._2,
        rangeCols.indices.map(l => {
          (0 until KVec(l)).map(k_l => {
            rangeCols(l).map(col => {
              this.gaussians(l)(k_l).logpdf(Vectors.dense(row._2(col).toArray))
            }).sum + logCondPiRows(l)(k_l)
          }).toList
        }).toList)
    })
  }

  def drawColPartition(data: RDD[(Int, Array[DenseVector[Double]], List[Int])]): List[Int] = {

    val jointLogDistribCols: List[List[Double]] = computeJointLogDistribColsFromSample(data)
    jointLogDistribCols.map(x => {
      val LSE = logSumExp(x)
      sample(x.map(e => exp(e - LSE)))
    })
  }

  def computeJointLogDistribColsFromSample(data: RDD[(Int, Array[DenseVector[Double]], List[Int])]): List[List[Double]] = {

    val logPiCols: DenseVector[Double] = DenseVector(this.proportionsCols.map(log(_)).toArray)
    var t0 = System.nanoTime()
    val D1 = data.flatMap(row => {
        row._2.indices.map(j => {
          (j, DenseVector(KVec.indices.map(l => {
            this.gaussians(l)(row._3(l))
              .logpdf(Vectors.dense(row._2(j).toArray))
          }).toArray))
        })
      })

    val D = D1.reduceByKey(_+_).collect().sortBy(_._1).map(_._2)
    val sumProb = D.map(e => (e+ logPiCols).toArray.toList).toList
    sumProb
  }

  def SEMGibbsMaximizationStep(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                               colPartition: List[Int],
                               n:Int,
                               fullCovariance: Boolean,
                               verbose: Boolean = true)(implicit ss: SparkSession): CondLatentBlockModel = {

    require(colPartition.distinct.length == KVec.length)
    val partitionPerColBc = ss.sparkContext.broadcast(DenseVector(colPartition:_*))
    val (means, covMat, sizeBlock, _) = getMeansAndCovariances(data, partitionPerColBc, KVec, fullCovariance, verbose)
    val Models: List[List[MultivariateGaussian]] =
      KVec.indices.map(l => {
        (0 until KVec(l)).map(k => {
          new MultivariateGaussian(
            Vectors.dense(means(l)(k).toArray),
            denseMatrixToMatrix(covMat(l)(k)))
        }).toList
      }).toList

    val proportions = proportionFromCLBMSizeBlock(sizeBlock)
    CondLatentBlockModel(proportions._1, proportions._2, Models)
  }

  def completelogLikelihood(dataWithRowPartition: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                            colPartition: List[Int]): Double = {

    val logRho: List[Double] = proportionsCols.map(log(_))
    val logPi: List[List[Double]]  = proportionsRows.map(_.map(log(_)))
    val rangeCols: List[List[Int]] = KVec.indices.map(l => colPartition.zipWithIndex.filter(_._1 == l).map(_._2)).toList

    dataWithRowPartition.map(row => {
      rangeCols.indices.map(l => {
        rangeCols(l).map(j => {
          logPi(l)(row._3(l))
          + logRho(l)
          + this.gaussians(l)(row._3(l)).logpdf(Vectors.dense(row._2(j).toArray))
        }).sum
      }).sum
    }).sum()
  }

  def ICL(completelogLikelihood: Double,
          n: Double,
          p: Double,
          fullCovariance: Boolean): Double = {

    val dimVar = this.gaussians.head.head.mean.size
    val L = KVec.length
    val nParamPerComponent = if(fullCovariance){
      dimVar+ dimVar*(dimVar+1)/2D
    } else {2D * dimVar}

    val nClusterRow = this.KVec.sum

    (completelogLikelihood
      - log(n)*(nClusterRow - L)/2D
      - log(p)*(L-1)/2D
      - log(n*p)*(nClusterRow*L*nParamPerComponent)/2D)
  }

}

