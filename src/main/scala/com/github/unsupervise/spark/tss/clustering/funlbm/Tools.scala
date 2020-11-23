/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.funlbm

import com.github.unsupervise.spark.tss.core.IO.{addPrefix, writeCsvFile}
import breeze.linalg.eigSym.EigSym
import breeze.linalg.{DenseMatrix, DenseVector, diag, eigSym, min, sum, svd}
import breeze.numerics.sqrt
import breeze.stats.distributions.RandBasis
import org.apache.spark.rdd.RDD

import scala.util.Try

object Tools extends java.io.Serializable {
  implicit val basis: RandBasis = RandBasis.withSeed(2)

  def writeFunLBMResults(SEMResult: Map[String, Product],
                         data: RDD[(Int, Array[DenseVector[Double]])],
                         series: RDD[(Int, Array[DenseVector[Double]])],
                         dictionaryId: Option[scala.collection.Map[String, Int]] = None,
                         dictionaryVarName: Option[scala.collection.Map[String, Int]],
                         pathOutput: String): Try[Unit] = {

    val outputModelPath = pathOutput+"model.csv"
    val outputDataPath = pathOutput+"dataAndCluster.csv"

    writeFunLBMParameters(SEMResult, outputModelPath)
    writeFunLBMData(data, series, SEMResult, outputDataPath, dictionaryId, dictionaryVarName)
  }

  def writeFunLBMParameters(SEMGibbsOutput: Map[String, Product], pathOutput: String)={

    val outputModel = SEMGibbsOutput("Model").asInstanceOf[FunLatentBlockModel]
    val outputContent = outputModel.proportionsCols.indices.map(l => {
      outputModel.proportionsRows.indices.map(k => {
        List(k.toString,
          l.toString,
          outputModel.proportionsRows(k).toString,
          outputModel.proportionsCols(l).toString,
          outputModel.gaussians(l)(k).mean.toArray.mkString(":"),
          outputModel.gaussians(l)(k).cov.toArray.mkString(":"),
          outputModel.loadings(l)(k).toArray.mkString(":"),
          outputModel.centerings(l)(k).toArray.mkString(":"))
      }).toList
    }).reduce(_++_)
    val header: List[String] = List("id", "rowCluster", "colCluster", "rowProportion", "colProportion",
      "meanListDV", "cov", "loadings", "centerings")
    writeCsvFile(pathOutput, addPrefix(outputContent),header)
  }

  def writeFunLBMData(data: RDD[(Int, Array[DenseVector[Double]])],
                      series: RDD[(Int, Array[DenseVector[Double]])],
                      SEMGibbsOutput: Map[String, Product],
                      pathOutput: String,
                      dictionaryId: Option[scala.collection.Map[String, Int]] = None,
                      dictionaryVarName: Option[scala.collection.Map[String, Int]] = None): Try[Unit] = {

    val rowPartition = SEMGibbsOutput("RowPartition").asInstanceOf[List[Int]]
    val colPartition = SEMGibbsOutput("ColPartition").asInstanceOf[List[Int]]

    val actualDictionaryId      = dictionaryId     .getOrElse(rowPartition.indices.map(_.toString).zipWithIndex.toMap)
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


  def projectInLowerSubspace(data: RDD[(Int, Array[DenseVector[Double]], Int)],
                             colPartition: List[Int],
                             loadings: List[List[DenseMatrix[Double]]],
                             centerings: List[List[DenseVector[Double]]]):
  RDD[(Int, Array[DenseVector[Double]], Int)] = {
    data.map(r => (r._1, r._2.indices.toArray.map(j => {
      val l = colPartition(j)
      loadings(l)(r._3) * (r._2(j)- centerings(l)(r._3))
    }), r._3))
  }

  def getLoadings(covarianceMatrix: DenseMatrix[Double],
                  nMaxEigenValues: Int= 3,
                  thresholdVarExplained: Double=0.99): DenseMatrix[Double] = {

    //    val currentMaxEigenValues = min()
    val EigSym(eVal, eVec) = eigSym(covarianceMatrix)

    val sortedEigVal = DenseVector(eVal.toArray.sorted)
    val sortedEigVec = DenseMatrix((0 until eVec.rows).map(i => (eVec(::,i), eVal(i))).sortBy(-_._2).map(_._1):_*)

    val normalizedEigenValues = eVal/sum(eVal)
    val idxMaxVarianceExplained1 = normalizedEigenValues.toArray.map{var s = 0D; d => {s += d; s}}
    val idxMaxVarianceExplained = idxMaxVarianceExplained1.toList.indexWhere(_> thresholdVarExplained)
    //    val idxKept = min(idxMaxVarianceExplained, nMaxEigenValues)
    val idxKept = min(idxMaxVarianceExplained, nMaxEigenValues - 1)//nMaxEigenValues-1
    val keptEigenVectors = sortedEigVec.t(::, 0 until idxKept)

    keptEigenVectors.t
  }

}
