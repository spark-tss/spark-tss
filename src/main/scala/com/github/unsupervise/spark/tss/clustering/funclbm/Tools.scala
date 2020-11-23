/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.funclbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
//import CLBMSpark.CondLatentBlockModel
import com.github.unsupervise.spark.tss.core.IO._
import breeze.linalg.{DenseMatrix, DenseVector, max}
import breeze.stats.distributions.RandBasis
import com.github.unsupervise.spark.tss.core.TSS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.col

import scala.io.Source
import scala.util.{Success, Try}

object Tools extends java.io.Serializable {
  implicit val basis: RandBasis = RandBasis.withSeed(2)

  def projectInLowerSubspace(data: RDD[(Int, Array[DenseVector[Double]], List[Int])],
                             colPartition: List[Int],
                             loadings: List[List[DenseMatrix[Double]]],
                             centerings: List[List[DenseVector[Double]]]):
  RDD[(Int, Array[DenseVector[Double]], List[Int])] = {
    data.map(r => (r._1, r._2.indices.toArray.map(j => {
      val l = colPartition(j)
      loadings(l)(r._3(l)) * (r._2(j)- centerings(l)(r._3(l)))
    }), r._3))
  }

  def printModel(model: FunCondLatentBlockModel): Unit={
    println("> Row proportions:")
    model.proportionsRows.foreach(println)
    println("> Column proportions:")
    println(model.proportionsCols)
    println("> Components Parameters")
    model.gaussians.foreach(m => m.foreach(s => println(s.mean, s.cov)))
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
    Tools.writeFunCLBMData(data, series , SEMResult, outputDataPath, dictionaryId, dictionaryVarName)
  }

  def writeFunCLBMParameters(SEMGibbsOutput: Map[String, Product], pathOutput: String)={

    val outputModel = SEMGibbsOutput("Model").asInstanceOf[FunCondLatentBlockModel]
    val gaussians = outputModel.gaussians
    val outputContent = outputModel.proportionsCols.indices.map(l => {
      outputModel.proportionsRows(l).indices.map(k_l => {
        List(k_l.toString,
          l.toString,
          outputModel.proportionsRows(l)(k_l).toString,
          outputModel.proportionsCols(l).toString,
          outputModel.gaussians(l)(k_l).mean.toArray.mkString(":"),
          outputModel.gaussians(l)(k_l).cov.toArray.mkString(":"),
          outputModel.loadings(l)(k_l).toArray.mkString(":"),
          outputModel.centerings(l)(k_l).toArray.mkString(":"))
      }).toList
    }).reduce(_++_)
    val header: List[String] = List("id", "rowCluster", "colCluster", "rowProportion", "colProportion",
      "meanListDV", "cov", "loadings", "centerings")
    writeCsvFile(pathOutput, addPrefix(outputContent),header)
  }

  def writeFunCLBMData(data: RDD[(Int, Array[DenseVector[Double]])],
                       series: RDD[(Int, Array[DenseVector[Double]])],
                       SEMGibbsOutput: Map[String, Product],
                       pathOutput: String,
                       dictionaryId: Option[scala.collection.Map[String, Int]] = None,
                       dictionaryVarName: Option[scala.collection.Map[String, Int]] = None): Try[Unit] ={

    val rowPartition = SEMGibbsOutput("RowPartition").asInstanceOf[List[List[Int]]]
    val colPartition = SEMGibbsOutput("ColPartition").asInstanceOf[List[Int]]

    val actualDictionaryId      = dictionaryId     .getOrElse(rowPartition.transpose.indices.map(_.toString).zipWithIndex.toMap)
    val actualDictionaryVarName = dictionaryVarName.getOrElse(colPartition.indices.map(_.toString).zipWithIndex.toMap)

    val reverseMapIterationId = for ((k, v) <- actualDictionaryId) yield (v, k)
    val reverseMapVarName = for ((k, v) <- actualDictionaryVarName) yield (v, k)

    val dataWithSeries = data.join(series).map(r =>
      (r._1, r._2._1, r._2._2)
    )

    val outputContent = dataWithSeries.flatMap(row => {
      row._2.indices.map(j => {
        val scenarioId = row._1 // Int Index
        val varId = j           // Int Index
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
