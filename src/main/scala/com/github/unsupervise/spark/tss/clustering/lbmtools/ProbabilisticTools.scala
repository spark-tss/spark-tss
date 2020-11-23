/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.lbmtools

import Tools._
import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, diag, max, sum}
import breeze.numerics.{exp, log}
import breeze.stats.distributions.RandBasis
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD

import scala.collection.mutable

object ProbabilisticTools extends java.io.Serializable {
  implicit val basis: RandBasis = RandBasis.withSeed(2)

  def variance(X: DenseVector[Double]): Double = {
    covariance(X,X)
  }

  def covariance(X: DenseVector[Double],Y: DenseVector[Double]): Double = {
    sum( (X- meanDV(X)) *:* (Y- meanDV(Y)) ) / (Y.length-1)
  }

  def covarianceSpark(X: RDD[((Int, Int), Vector)],
                      modes: Map[(Int, Int), DenseVector[Double]],
                      count: Map[(Int, Int), Int]): Map[(Int, Int),  DenseMatrix[Double]] = {

    val XCentered : RDD[((Int, Int), DenseVector[Double])] = X.map(d => (d._1, DenseVector(d._2.toArray) - DenseVector(modes(d._1).toArray)))

    val internProduct = XCentered.map(row => (row._1, row._2 * row._2.t))
    val internProductSumRDD: RDD[((Int,Int), DenseMatrix[Double])] = internProduct.reduceByKey(_+_)
    val interProductSumList: List[((Int,Int),  DenseMatrix[Double])] = internProductSumRDD.collect().toList

    interProductSumList.map(c => (c._1,c._2/(count(c._1)-1).toDouble)).toMap

  }

  def covariance(X: List[DenseVector[Double]], mode: DenseVector[Double], constraint: String = "none"): DenseMatrix[Double] = {

    require(List("none","independant").contains(constraint))
    require(mode.length==X.head.length)
    val XMat: DenseMatrix[Double] = DenseMatrix(X.toArray:_*)
    val p = XMat.cols
    constraint match {
      case "independant" => DenseMatrix.tabulate[Double](p,p){(i, j) => if(i == j) covariance(XMat(::,i),XMat(::,i)) else 0D}
      case _ => {
        val modeMat: DenseMatrix[Double] = DenseMatrix.ones[Double](X.length,1) * mode.t
        val XMatCentered: DenseMatrix[Double] = XMat - modeMat
        XMatCentered.t * XMatCentered
      }/ (X.length.toDouble)
    }
  }

  def weightedCovariance(X: DenseVector[Double], Y: DenseVector[Double], weights: DenseVector[Double]): Double = {
    sum( weights *:* (X- meanDV(X)) *:* (Y- meanDV(Y)) ) / sum(weights)
  }

  def weightedCovariance (X: List[DenseVector[Double]],
                          weights: DenseVector[Double],
                          mode: DenseVector[Double],
                          constraint: String = "none"): DenseMatrix[Double] = {
    require(List("none","independant").contains(constraint))
    require(mode.length==X.head.length)
    require(weights.length==X.length)

    val XMat: DenseMatrix[Double] = DenseMatrix(X.toArray:_*)
//    val p = XMat.cols
    val q = mode.length
    constraint match {
      case "independant" => DenseMatrix.tabulate[Double](q,q){(i, j) => if(i == j) weightedCovariance(XMat(::,i),XMat(::,i), weights) else 0D}
      case _ =>
        val modeMat: DenseMatrix[Double] = DenseMatrix.ones[Double](X.length,1) * mode.t
        val XMatCentered: DenseMatrix[Double] = XMat - modeMat
        val res = DenseMatrix((0 until XMatCentered.rows).par.map(i => {
          weights(i)*XMatCentered(i,::).t * XMatCentered(i,::)
        }).reduce(_+_).toArray:_*)/ sum(weights)
        res.reshape(q,q)
    }
  }

  def meanDV(X: DenseVector[Double]): Double = {
    sum(X)/X.length
  }

  def meanListDV(X: List[DenseVector[Double]]): DenseVector[Double] = {
    require(X.nonEmpty)
    X.reduce(_+_) / X.length.toDouble
  }

  def weightedMean(X: List[DenseVector[Double]], weights: DenseVector[Double]): DenseVector[Double] = {
    require(X.length == weights.length)
    val res = X.indices.par.map(i => weights(i) * X(i)).reduce(_+_) / sum(weights)
    res
  }

  def sample(probabilities: List[Double]): Int = {
    val dist = probabilities.indices zip probabilities
    val threshold = scala.util.Random.nextDouble
    val iterator = dist.iterator
    var accumulator = 0.0
    while (iterator.hasNext) {
      val (cluster, clusterProb) = iterator.next
      accumulator += clusterProb
      if (accumulator >= threshold)
        return cluster
    }
    sys.error("Error")
  }

  def intToVecProb(i: Int, size:Int): List[Double] = {
    val b = mutable.Buffer.fill(size)(1e-8)
    b(i)=1D
    val sum = b.sum
    b.map(_/sum) .toList
  }

  def partitionToBelongingProbabilities(partition: List[Int], toLog:Boolean=false): List[List[Double]]={

    val K = partition.max+1
    val res = partition.indices.map(i => {
      intToVecProb(partition(i),K)
    }).toList

    if(!toLog){res} else {res.map(_.map(log(_)))}
  }


  def logSumExp(X: List[Double]): Double ={
    val maxValue = max(X)
    maxValue + log(sum(X.map(x => exp(x-maxValue))))
  }

  def logSumExp(X: DenseVector[Double]): Double ={
    val maxValue = max(X)
    maxValue + log(sum(X.map(x => exp(x-maxValue))))
  }

  def unitCovFunc(fullCovarianceMatrix:Boolean):
  DenseVector[Double] => DenseMatrix[Double] = (x: DenseVector[Double]) => {
    if(fullCovarianceMatrix){
      x * x.t
    } else {
      diag(x *:* x)
    }
  }

  def mapDm(probBelonging: DenseMatrix[Double]): List[Int] = {
    probBelonging(*,::).map(argmax(_)).toArray.toList
  }

  def MAP(probBelonging: List[List[Double]]): List[Int] = {
    probBelonging.map(Tools.argmax)
  }
}
