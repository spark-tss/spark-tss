/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.clbm

import com.github.unsupervise.spark.tss.core.IO._
import com.github.unsupervise.spark.tss.clustering.lbmtools.ProbabilisticTools.unitCovFunc
import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools.{remove, prettyPrint}
import breeze.linalg.DenseVector
import breeze.stats.distributions.RandBasis
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.util.Try

object Tools extends java.io.Serializable {
  implicit val basis: RandBasis = RandBasis.withSeed(2)

  def getRowPartitionFromDataWithRowPartition(precData: RDD[(Int, Array[DenseVector[Double]], List[Int])]) = {
    precData.map(row => (row._1, row._3))
      .collect().sortBy(_._1).map(_._2).toList.transpose
  }

  def getSizeAndSumByBlock(data: RDD[((Int, Int), Array[DenseVector[Double]])]) = {
    data
      .map(r => (r._1, (r._2.reduce(_+_), r._2.length)))
      .reduceByKey((a,b) => (a._1 + b._1, a._2+b._2))
  }

  private def getDataPerColumnAndRow(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                                     partitionPerColBc: Broadcast[DenseVector[Int]],
                                     KVec: List[Int]) = {
    val dataPerColumnAndRow = data.flatMap(row => {
      KVec.indices.map(l => {
        val rowDv = DenseVector(row._2)
        val dataColumn = rowDv(partitionPerColBc.value:==l)
        ((l, row._3(l)), dataColumn.toArray)
      })
    }).cache()
    dataPerColumnAndRow
  }

  private def getCovarianceMatrices(dataPerColumnAndRow: RDD[((Int, Int), Array[DenseVector[Double]])],
                                    meanByBlock: Map[(Int, Int), DenseVector[Double]],
                                    sizeBlockMap: Map[(Int, Int), Int],
                                    KVec: List[Int],
                                    fullCovariance: Boolean=true) = {
    val covFunction = unitCovFunc(fullCovariance)

    val sumCentered = dataPerColumnAndRow.map(r => {
      (r._1, r._2.map(v => covFunction(v - meanByBlock(r._1))).reduce(_+_))
    }).reduceByKey(_+_).collect().toMap
    KVec.indices.map(l => {
      (0 until KVec(l)).map(k => {
        sumCentered(l,k)/(sizeBlockMap(l,k).toDouble-1)
      })
    })
  }

  def getMeansAndCovariances(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                             colPartition: Broadcast[DenseVector[Int]],
                             KVec: List[Int],
                             fullCovariance: Boolean,
                             verbose: Boolean) = {
    val dataPerColumnAndRow: RDD[((Int, Int), Array[DenseVector[Double]])] =
      getDataPerColumnAndRow(data, colPartition, KVec)
    val sizeAndSumBlock = getSizeAndSumByBlock(dataPerColumnAndRow)
    val sizeBlock = sizeAndSumBlock.map(r => (r._1, r._2._2))
    val sizeBlockMap = sizeBlock.collect().toMap
    if(verbose){prettyPrint(sizeBlockMap)}
    val meanByBlock: Map[(Int, Int), DenseVector[Double]] = sizeAndSumBlock.map(r => (r._1, r._2._1 / r._2._2.toDouble)).collect().toMap
    val covMat = getCovarianceMatrices(dataPerColumnAndRow, meanByBlock, sizeBlockMap, KVec,fullCovariance)
    val listMeans = KVec.indices.map(l => {
      (0 until KVec(l)).map(k => {
        meanByBlock(l, k)
      }).toList
    }).toList
    (listMeans, covMat, sizeBlock, sizeBlockMap)
  }

  def printModel(model: CondLatentBlockModel): Unit={
    println("> Row proportions:")
    model.proportionsRows.foreach(println)
    println("> Column proportions:")
    println(model.proportionsCols)
    println("> Components Parameters")
    model.gaussians.foreach(m => m.foreach(s => println(s.mean, s.cov)))
  }

  def updateModel(formerModel: CondLatentBlockModel,
                  colClusterToUpdate: Int,
                  modelToInsert:CondLatentBlockModel): CondLatentBlockModel = {

    require(colClusterToUpdate >= 0 & colClusterToUpdate<formerModel.proportionsCols.length,
      "Col Cluster Idx to replace should be > 0 and lesser than the column cluster number of former model")

    val newRowProportion = remove(formerModel.proportionsRows, colClusterToUpdate) ++ modelToInsert.proportionsRows
    val newColProportion = remove(formerModel.proportionsCols, colClusterToUpdate) ++
      modelToInsert.proportionsCols.map(c => c*formerModel.proportionsCols(colClusterToUpdate))
    val newModels = remove(formerModel.gaussians, colClusterToUpdate) ++ modelToInsert.gaussians

    CondLatentBlockModel(newRowProportion, newColProportion, newModels)
  }

  def writeCLBMResults(SEMResult: Map[String, Product],
                       data: RDD[(Int, Array[DenseVector[Double]])],
                       dictionaryId: Option[scala.collection.Map[String, Int]] = None,
                       dictionaryVarName: Option[scala.collection.Map[String, Int]],
                       pathOutput: String): Try[Unit] = {

    val outputModelPath = pathOutput+"model.csv"
    val outputDataPath = pathOutput+"dataAndCluster.csv"

    Tools.writeCLBMParameters(SEMResult, outputModelPath)
    Tools.writeCLBMData(data, SEMResult, outputDataPath, dictionaryId, dictionaryVarName)
  }

  def writeCLBMParameters(SEMGibbsOutput: Map[String, Product], pathOutput: String): Try[Unit] ={

    val outputModel = SEMGibbsOutput("Model").asInstanceOf[CondLatentBlockModel]
    val outputContent = outputModel.proportionsCols.indices.map(l => {
      outputModel.proportionsRows(l).indices.map(k_l => {
        List(k_l.toString,
          l.toString,
          outputModel.proportionsRows(l)(k_l).toString,
          outputModel.proportionsCols(l).toString,
          outputModel.gaussians(l)(k_l).mean.toArray.mkString(":"),
          outputModel.gaussians(l)(k_l).cov.toArray.mkString(":"))
      }).toList
    }).reduce(_++_)

    val header: List[String] = List("id","rowCluster","colCluster","rowProportion", "colProportion", "meanListDV","cov")
    writeCsvFile(pathOutput, addPrefix(outputContent),header)
  }

  def writeCLBMData(dataRDD: RDD[(Int, Array[DenseVector[Double]])],
                       SEMGibbsOutput: Map[String, Product],
                       pathOutput: String,
                       dictionaryId: Option[scala.collection.Map[String, Int]] = None,
                       dictionaryVarName: Option[scala.collection.Map[String, Int]] = None): Try[Unit] ={

    val rowPartition = SEMGibbsOutput("RowPartition").asInstanceOf[List[List[Int]]]
    val colPartition = SEMGibbsOutput("ColPartition").asInstanceOf[List[Int]]
    val actualDictionaryId = dictionaryId.getOrElse(rowPartition.transpose.indices.map(_.toString).zipWithIndex.toMap)
    val actualDictionaryVarName = dictionaryVarName.getOrElse(colPartition.indices.map(_.toString).zipWithIndex.toMap)
    val reverseMapIterationId = for ((k,v) <- actualDictionaryId) yield (v, k)
    val reverseMapVarName = for ((k,v) <- actualDictionaryVarName) yield (v, k)

    val outputContent = dataRDD.flatMap(row => {
      row._2.indices.map(j => {
        val scenarioId = row._1 // Int Index
        val varId = j        // Int Index
        val l = colPartition(varId)
        val k_l = rowPartition(l)(scenarioId)
        List(
          reverseMapIterationId(scenarioId),
          reverseMapVarName(varId),
          k_l.toString,
          l.toString,
          row._2(varId).toArray.mkString(":")
        )
      })
    }).collect().toList

    writeCsvFile(pathOutput,outputContent)
  }

  def writeFunCLBMResults(SEMResult: Map[String, Product],
                          data: RDD[(Int, Array[DenseVector[Double]])],
                          series: RDD[(Int, Array[DenseVector[Double]])],
                          dictionaryId: Option[scala.collection.Map[String, Int]] = None,
                          dictionaryVarName: Option[scala.collection.Map[String, Int]],
                          pathOutput: String): Try[Unit] = {

    val outputModelPath = pathOutput+"model.csv"
    val outputDataPath = pathOutput+"dataAndCluster.csv"

    Tools.writeFunCLBMParameters(SEMResult, outputModelPath)
    Tools.writeFunCLBMData(data, series, SEMResult, outputDataPath, dictionaryId, dictionaryVarName)
  }

  def writeFunCLBMParameters(SEMGibbsOutput: Map[String, Product], pathOutput: String): Try[Unit] ={

    val outputModel = SEMGibbsOutput("Model").asInstanceOf[CondLatentBlockModel]
    val outputContent = outputModel.proportionsCols.indices.map(l => {
      outputModel.proportionsRows(l).indices.map(k_l => {
        List(k_l.toString,
          l.toString,
          outputModel.proportionsRows(l)(k_l).toString,
          outputModel.proportionsCols(l).toString,
          outputModel.gaussians(l)(k_l).mean.toArray.mkString(":"),
          outputModel.gaussians(l)(k_l).cov.toArray.mkString(":"))
      }).toList
    }).reduce(_++_)

    val header: List[String] = List("id","rowCluster","colCluster","rowProportion", "colProportion", "meanListDV","cov")
    writeCsvFile(pathOutput, addPrefix(outputContent),header)
  }

  def writeFunCLBMData(dataRDD: RDD[(Int, Array[DenseVector[Double]])],
                       series: RDD[(Int, Array[DenseVector[Double]])],
                       SEMGibbsOutput: Map[String, Product],
                       pathOutput: String,
                       dictionaryId: Option[scala.collection.Map[String, Int]] = None,
                       dictionaryVarName: Option[scala.collection.Map[String, Int]] = None): Try[Unit] ={

    val rowPartition = SEMGibbsOutput("RowPartition").asInstanceOf[List[List[Int]]]
    val colPartition = SEMGibbsOutput("ColPartition").asInstanceOf[List[Int]]
    val actualDictionaryId = dictionaryId.getOrElse(rowPartition.transpose.indices.map(_.toString).zipWithIndex.toMap)
    val actualDictionaryVarName = dictionaryVarName.getOrElse(colPartition.indices.map(_.toString).zipWithIndex.toMap)
    val reverseMapIterationId = for ((k,v) <- actualDictionaryId) yield (v, k)
    val reverseMapVarName = for ((k,v) <- actualDictionaryVarName) yield (v, k)
    val dataWithSeries = dataRDD.join(series).map(r => (r._1, r._2._1, r._2._2))

    val outputContent = dataWithSeries.flatMap(row => {
      row._2.indices.map(j => {
        val scenarioId = row._1 // Int Index
        val varId = j        // Int Index
        val l = colPartition(varId)
        val k_l = rowPartition(l)(scenarioId)
        List(
          reverseMapIterationId(scenarioId),
          reverseMapVarName(varId),
          k_l.toString,
          l.toString,
          row._2(varId).toArray.mkString(":"),
          row._3(varId).toArray.mkString(":")
        )
      })
    }).collect().toList

    writeCsvFile(pathOutput,outputContent)
  }


}
