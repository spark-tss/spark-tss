/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.clbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.ProbabilisticTools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import com.github.unsupervise.spark.tss.clustering.lbmtools.ToolsRDD._

import breeze.linalg.{DenseMatrix, DenseVector, argmax}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.distribution.MultivariateGaussian
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.Random

object Initialization  {

  def initialize(data: RDD[(Int, Array[DenseVector[Double]])],
                 condLatentBlock: CondLatentBlock,
                 EMMethod: String,
                 n:Int, p :Int,
                 verbose:Boolean = true,
                 initMethod: String = "randomPartition")(implicit ss: SparkSession): (CondLatentBlockModel, List[Int]) = {

    val KVec = condLatentBlock.getKVec
    val nSampleForLBMInit = 10
    initMethod match {
      case "random" =>
        val model = Initialization.randomModelInitialization(data, KVec, nSampleForLBMInit)
        (model, (0 until p).map(_ => sample(model.proportionsCols)).toList)
      case "randomPartition" => Initialization.initFromRandomPartition(data, KVec,n,p)
      case _ =>
        println(s"Warning: No initial method has been provided and initMethod $initMethod provided " +
          "does not match possible initialization method name (\"random\",\"sample\")" +
          "Continuing with random initialization..")
        val model = Initialization.randomModelInitialization(data,KVec,nSampleForLBMInit)
        (model, (0 until p).map(_ => sample(model.proportionsCols)).toList)
    }
  }
  def randomModelInitialization(data: RDD[(Int, Array[DenseVector[Double]])],
                                KVec: List[Int],
                                nSamples:Int = 10)(implicit ss: SparkSession): CondLatentBlockModel = {

    val L = KVec.length
    val MultivariateGaussians: List[List[MultivariateGaussian]] = KVec.indices.map(l => {
      (0 until KVec(l)).map(_ => {
        val sampleBlock: List[DenseVector[Double]] = data.takeSample(withReplacement = false, nSamples)
          .map(e => Random.shuffle(e._2.toList).head).toList
        val mode: DenseVector[Double] = meanListDV(sampleBlock)
        new MultivariateGaussian(Vectors.dense(mode.toArray), denseMatrixToMatrix(
          covariance(sampleBlock, mode)))
      }).toList
    }).toList

    val rowProportions:List[List[Double]] = (0 until L).map(l => {List.fill(KVec(l))(1.0 / KVec(l))}).toList
    val colProportions:List[Double] =  List.fill(L)(1.0 / L):List[Double]
    CondLatentBlockModel(rowProportions, colProportions, MultivariateGaussians)
  }

  def sampleModelInitialization(data: RDD[(Int, Array[DenseVector[Double]])],
                                KVec: List[Int],
                                proportionSample:Double = 0.2,
                                nTry: Int = 5,
                                nConcurrentPerTry: Int = 3)(implicit ss: SparkSession): (CondLatentBlockModel, List[Int]) = {

    require(proportionSample>0 & proportionSample<=1, "proportionSample argument is a proportion (should be >0 and <=1)")

    val resList = (0 until nTry).map(_ => {
      val dataSample = data.sample(withReplacement = false, proportionSample)
      val CLBM = new CondLatentBlock().setKVec(KVec)
      CLBM.run(dataSample, nConcurrent=nConcurrentPerTry, nTryMaxPerConcurrent=10,initMethod = "random")
    })

    val allLikelihoods: DenseVector[Double] = DenseVector(resList.map(_("LogLikelihood").asInstanceOf[List[Double]].last).toArray)
    val bestRes = resList(argmax(allLikelihoods))

    (bestRes("Model").asInstanceOf[CondLatentBlockModel], bestRes("ColPartition").asInstanceOf[List[Int]])
  }

  def initFromRandomPartition(data: RDD[(Int, Array[DenseVector[Double]])],
                              KVec: List[Int],
                              n: Int,
                              p: Int,
                              verbose: Boolean=false)(implicit ss: SparkSession): (CondLatentBlockModel, List[Int]) = {
    if(verbose) println("Random Partition Initialization")
    val colPartition: List[Int] = Random.shuffle((0 until p).map(j => j%KVec.length)).toList


    val randomRowPartition: List[List[Int]] = KVec.indices.map(l => {
      Random.shuffle((0 until n).map(_%KVec(l))).toList
    }).toList

    val dataWithRowPartition = joinRowPartitionToData(data, randomRowPartition,n)

    initFromGivenPartition(dataWithRowPartition, colPartition, KVec, n)
  }

  def initFromGivenPartition(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                             colPartition: List[Int],
                             KVec: List[Int],
                             n: Int): (CondLatentBlockModel, List[Int]) = {

    val dataAndSizeByBlock =  KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {
        val filteredData = data.filter(_._3(l) == k_l).map(row => {row._2.zipWithIndex.filter(s => colPartition(s._2) == l).map(_._1)})
        val sizeBlock: Int = filteredData.map(_.length).sum().toInt
        require(sizeBlock > 0, "Algorithm could not converge: empty block")
        (filteredData, sizeBlock)
      })
    })

    val newModels: List[List[MultivariateGaussian]] =
      KVec.indices.map(l => {
        (0 until KVec(l)).map(k_l => {
          val filteredRDD = dataAndSizeByBlock(l)(k_l)._1
          val sizeBlock = dataAndSizeByBlock(l)(k_l)._2
          val mode: DenseVector[Double] = filteredRDD.map(_.reduce(_ + _)).reduce(_ + _) / (sizeBlock - 1).toDouble
          val covariance: DenseMatrix[Double] = filteredRDD.map(_.map(v => {
            val vc: DenseVector[Double] = v - mode
            vc * vc.t
          }).reduce(_ + _)).reduce(_ + _) / (sizeBlock - 1).toDouble
          val model: MultivariateGaussian = new MultivariateGaussian(
            Vectors.dense(mode.toArray),
            denseMatrixToMatrix(covariance))
          model
        }).toList
      }).toList

    val countRows: Map[(Int,Int), Int] = data.map(row => {
      KVec.indices.map(l => {(row._3(l),l)}).toList
    }).reduce(_ ++ _).groupBy(identity).mapValues(_.size)
    val proportionRows: List[List[Double]] = KVec.indices.map(l => {
      (0 until KVec(l)).map(k_l => {countRows(k_l,l)/n.toDouble}).toList
    }).toList

    val countCols = colPartition.groupBy(identity).mapValues(_.size)
    val proportionCols = countCols.map(c => c._2 / countCols.values.sum.toDouble).toList

    (CondLatentBlockModel(proportionRows, proportionCols, newModels), colPartition)
  }
}

