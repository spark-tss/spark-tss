/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.lbm

import com.github.unsupervise.spark.tss.clustering.clbm.Tools._
import com.github.unsupervise.spark.tss.core.IO._
import com.github.unsupervise.spark.tss.clustering.lbmtools.ProbabilisticTools.unitCovFunc
import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools.prettyPrint
import breeze.linalg.{DenseMatrix, DenseVector, dim}
import com.github.unsupervise.spark.tss.core.TSS
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import com.github.unsupervise.spark.tss.clustering.lbmtools.ToolsRDD._

import scala.util.Try

object Tools extends java.io.Serializable {

  def getRowPartitionFromDataWithRowPartition(precData: RDD[(Int, Array[DenseVector[Double]], List[Int])]) = {
    precData.map(row => (row._1, row._3))
      .collect().sortBy(_._1).map(_._2).toList.transpose
  }

  def getSizeAndSumByBlock(data: RDD[((Int, Int), Array[DenseVector[Double]])]) = {
    data
      .map(r => (r._1, (r._2.reduce(_+_), r._2.length)))
      .reduceByKey((a,b) => (a._1 + b._1, a._2+b._2))
  }

  private def getDataPerColumnAndRow(periodogram: RDD[(Int, Array[DenseVector[Double]], Int)],
                                     partitionPerColBc: Broadcast[DenseVector[Int]],
                                     L:Int) = {

    val dataPerColumnAndRow: RDD[((Int, Int), Array[DenseVector[Double]])] = periodogram.flatMap(row => {
      (0 until L).map(l => {
        val rowDv = DenseVector(row._2)
        val cells = rowDv(partitionPerColBc.value :== l)
        ((l, row._3), cells.toArray)
      })
    }).cache()
    dataPerColumnAndRow
  }

  private def getCovarianceMatrices(dataPerColumnAndRow: RDD[((Int, Int), Array[DenseVector[Double]])],
                                    meanByBlock: Map[(Int, Int), DenseVector[Double]],
                                    sizeBlockMap: Map[(Int, Int), Int],
                                    L:Int,
                                    K:Int,
                                    fullCovariance: Boolean) = {
    val covFunction = unitCovFunc(fullCovariance)

    val sumCentered = dataPerColumnAndRow.map(r => {
      (r._1, r._2.map(v => covFunction(v - meanByBlock(r._1))).reduce(_ + _))
    }).reduceByKey(_ + _).collect().toMap
    val res = (0 until L).map(l => {
      (0 until K).map(k => {
        sumCentered(l, k) / (sizeBlockMap(l, k).toDouble - 1)
      }).toList
    }).toList
    res

  }

  def getMeansAndCovariances(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                             colPartition: Broadcast[DenseVector[Int]],
                             L:Int,
                             K:Int,
                             fullCovariance: Boolean) = {
    val dataPerColumnAndRow: RDD[((Int, Int), Array[DenseVector[Double]])] = getDataPerColumnAndRow(data, colPartition, L)
    val sizeAndSumBlock = getSizeAndSumByBlock(dataPerColumnAndRow)
    val sizeBlock = sizeAndSumBlock.map(r => (r._1, r._2._2))
    val sizeBlockMap = sizeBlock.collect().toMap
    prettyPrint(sizeBlockMap)
    val meanByBlock: Map[(Int, Int), DenseVector[Double]] = sizeAndSumBlock.map(r => (r._1, r._2._1 / r._2._2.toDouble)).collect().toMap
    val covMat = getCovarianceMatrices(dataPerColumnAndRow, meanByBlock, sizeBlockMap, L, K, fullCovariance)
    val listMeans = (0 until L).map(l => {
      (0 until K).map(k => {
        meanByBlock(l, k)
      }).toList
    }).toList
    (listMeans, covMat, sizeBlock, sizeBlockMap)
  }

  def writeLBMResults(SEMResult: Map[String, Product],
                      data: RDD[(Int, Array[DenseVector[Double]])],
                      dictionaryId: Option[scala.collection.Map[String, Int]] = None,
                      dictionaryVarName: Option[scala.collection.Map[String, Int]],
                      pathOutput: String): Try[Unit] = {

    val outputModelPath = pathOutput+"model.csv"
    val outputDataPath = pathOutput+"dataAndCluster.csv"

    Tools.writeLBMParameters(SEMResult, outputModelPath)
    Tools.writeLBMData(data, SEMResult, outputDataPath, dictionaryId, dictionaryVarName)
  }

  def writeLBMParameters(SEMGibbsOutput: Map[String, Product], pathOutput: String)={

    val outputModel = SEMGibbsOutput("Model").asInstanceOf[LatentBlockModel]
    val gaussians = outputModel.gaussians
    val outputContent =
      outputModel.proportionsCols.indices.map(l => {
        outputModel.proportionsRows.indices.map(k => {
          List(k.toString,
            l.toString,
            outputModel.proportionsRows(k).toString,
            outputModel.proportionsCols(l).toString,
            gaussians(l)(k).mean.toArray.mkString(":"),
            gaussians(l)(k).cov.toArray.mkString(":"))
        }).toList
      }).reduce(_++_)

    val header: List[String] = List("id","rowCluster","colCluster","rowProportion", "colProportion", "meanListDV","cov")
    writeCsvFile(pathOutput, addPrefix(outputContent),header)
  }

  def writeLBMData(dataRDD: RDD[(Int, Array[DenseVector[Double]])],
                   SEMGibbsOutput: Map[String, Product],
                   pathOutput: String,
                   dictionaryId: Option[scala.collection.Map[String, Int]] = None,
                   dictionaryVarName: Option[scala.collection.Map[String, Int]] = None): Try[Unit] = {

    val rowPartition = SEMGibbsOutput("RowPartition").asInstanceOf[List[Int]]
    val colPartition = SEMGibbsOutput("ColPartition").asInstanceOf[List[Int]]

    val actualDictionaryId = dictionaryId.getOrElse(rowPartition.indices.map(_.toString).zipWithIndex.toMap)
    val actualDictionaryVarName = dictionaryVarName.getOrElse(colPartition.indices.map(_.toString).zipWithIndex.toMap)

    val reverseMapIterationId = for ((k, v) <- actualDictionaryId) yield (v, k)
    val reverseMapVarName = for ((k, v) <- actualDictionaryVarName) yield (v, k)

    val outputContent = dataRDD.flatMap(row => {
      row._2.indices.map(j => {
        val scenarioId = row._1 // Int Index
        val varId = j // Int Index
        val l = colPartition(varId)
        val k = rowPartition(scenarioId)
        List(
          reverseMapIterationId(scenarioId),
          reverseMapVarName(varId),
          k.toString,
          l.toString,
          row._2(varId).toArray.mkString(":"))
      })
    }).collect().toList

    writeCsvFile(pathOutput, outputContent)
  }

  def writeFunLBMResults(SEMResult: Map[String, Product],
                         data: RDD[(Int, Array[DenseVector[Double]])],
                         series: RDD[(Int, Array[DenseVector[Double]])],
                         dictionaryId: Option[scala.collection.Map[String, Int]] = None,
                         dictionaryVarName: Option[scala.collection.Map[String, Int]],
                         pathOutput: String): Try[Unit] = {

    val outputModelPath = pathOutput + "model.csv"
    val outputDataPath  = pathOutput + "dataAndCluster.csv"

    Tools.writeFunLBMParameters(SEMResult, outputModelPath)
    Tools.writeFunLBMData(data, series, SEMResult, outputDataPath, dictionaryId, dictionaryVarName)
  }

  def writeFunLBMParameters(SEMGibbsOutput: Map[String, Product], pathOutput: String)={

    val outputModel = SEMGibbsOutput("Model").asInstanceOf[LatentBlockModel]
    val gaussians = outputModel.gaussians
    val outputContent =
      outputModel.proportionsCols.indices.map(l => {
        outputModel.proportionsRows.indices.map(k => {
          List(k.toString,
            l.toString,
            outputModel.proportionsRows(k).toString,
            outputModel.proportionsCols(l).toString,
            gaussians(l)(k).mean.toArray.mkString(":"),
            gaussians(l)(k).cov.toArray.mkString(":"))
        }).toList
      }).reduce(_++_)

    val header: List[String] = List("id","rowCluster","colCluster","rowProportion", "colProportion", "meanListDV","cov")
    writeCsvFile(pathOutput, addPrefix(outputContent),header)
  }

  def writeFunLBMData(dataRDD: RDD[(Int, Array[DenseVector[Double]])],
                      series: RDD[(Int, Array[DenseVector[Double]])],
                      SEMGibbsOutput: Map[String, Product],
                      pathOutput: String,
                      dictionaryId: Option[scala.collection.Map[String, Int]] = None,
                      dictionaryVarName: Option[scala.collection.Map[String, Int]] = None): Try[Unit] = {

    val rowPartition = SEMGibbsOutput("RowPartition").asInstanceOf[List[Int]]
    val colPartition = SEMGibbsOutput("ColPartition").asInstanceOf[List[Int]]
    val actualDictionaryId = dictionaryId.getOrElse(rowPartition.indices.map(_.toString).zipWithIndex.toMap)
    val actualDictionaryVarName = dictionaryVarName.getOrElse(colPartition.indices.map(_.toString).zipWithIndex.toMap)
    val reverseMapIterationId = for ((k, v) <- actualDictionaryId) yield (v, k)
    val reverseMapVarName = for ((k, v) <- actualDictionaryVarName) yield (v, k)

    val dataWithSeries = dataRDD.join(series).map(r =>
      (r._1, r._2._1, r._2._2)
    )

    val outputContent = dataWithSeries.flatMap(row => {
      row._2.indices.map(j => {
        val scenarioId = row._1 // Int Index
        val varId = j // Int Index
        val l = colPartition(varId)
        val k = rowPartition(scenarioId)
        List(
          reverseMapIterationId(scenarioId),
          reverseMapVarName(varId),
          k.toString,
          l.toString,
          row._2(varId).toArray.mkString(":"),
          row._3(varId).toArray.mkString(":")
        )
      })
    }).collect().toList
    writeCsvFile(pathOutput, outputContent)
  }

}
