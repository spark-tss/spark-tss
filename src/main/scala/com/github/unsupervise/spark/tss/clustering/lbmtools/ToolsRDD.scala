/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.lbmtools

import ProbabilisticTools._
import Tools._
import breeze.linalg.{DenseMatrix, DenseVector, argmax => bzArgMax}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object ToolsRDD extends java.io.Serializable {

  def fromCellDistributionToRowDistribution(data : RDD[(Int, Int, DenseVector[Double])])= {
    val dataWithIndex = data.map(row => (row._1,(row._2, row._3)))
    dataWithIndex.groupByKey.map(row =>
      (row._1, row._2.toList.sortBy(_._1).map(_._2).toArray))
  }

  def proportionFromCLBMSizeBlock(sizeBlock: RDD[((Int, Int), Int)]): (List[List[Double]], List[Double]) = {

    val sizeCol = sizeBlock.map(r => (r._1._1, r._2)).reduceByKey(_ + _).collect().sortBy(_._1).map(_._2).toList

    val p = sizeCol.sum
    val colProportions = sizeCol.map(_/p.toDouble)

    val rowProportions =
      colProportions.indices.map(l => {
        val sizeRow = sizeBlock.filter(_._1._1==l).map(r => (r._1._2, r._2)).collect().sortBy(_._1).map(_._2).toList
        val n = sizeRow.sum
        sizeRow.map(_/n.toDouble)
      }).toList
    (rowProportions, colProportions)
  }

  def proportionFromLBMSizeBlock(sizeBlock: RDD[((Int, Int), Int)]): (List[Double], List[Double]) = {

    val sizeCol = sizeBlock.map(r => (r._1._1, r._2)).reduceByKey(_ + _).collect().sortBy(_._1).map(_._2).toList
    val sizeRow = sizeBlock.map(r => (r._1._2, r._2)).reduceByKey(_ + _).collect().sortBy(_._1).map(_._2).toList

    val p = sizeCol.sum
    val n = sizeRow.sum

    val colProportions = sizeCol.map(_/p.toDouble)
    val rowProportions = sizeRow.map(_/n.toDouble)

    (rowProportions, colProportions)
  }

  def getCLBMRowPartitionFromDataWithRowPartition(d: RDD[(Int, Array[DenseVector[Double]], List[Int])]): List[List[Int]]={
    d.map(r => (r._1, r._3)).collect().sortBy(_._1).map(_._2).toList.transpose
  }

  def getRowPartitionFromDataWithRowPartition(d: RDD[(Int, Array[DenseVector[Double]], Int)]): List[Int]={
    d.map(r => (r._1, r._3)).collect().sortBy(_._1).map(_._2).toList
  }

  def getRowPartitionFromDataWithProbBelonging(d: RDD[(Int, Array[DenseVector[Double]], DenseVector[Double])]): List[Int]={
    val probs = d.map(r => (r._1, r._3)).collect().sortBy(_._1).map(_._2).toList
    probs.map(c => bzArgMax(c))
  }

  def RDDBlockToRow(data: RDD[((Int, Int), Vector)]): RDD[(Int, Array[Vector])] = {
    data
      .map(elt => (elt._1._1, (elt._1._2, elt._2)))
      .groupByKey()
      .map(elt => {(elt._1, elt._2.toArray.sortBy(_._1).map(_._2))})
  }

  def RDDBlockToCol(data: RDD[((Int, Int), Vector)]): RDD[(Int, Array[Vector])] = {
    data
      .map(elt => (elt._1._2, (elt._1._1, elt._2)))
      .groupByKey()
      .map(elt => {(elt._1, elt._2.toArray.sortBy(_._1).map(_._2))})
  }

  def dataWithProbFromDataWithPartitions(data: RDD[(Int, Int, DenseVector[Double], Int, Int)], K:Int, L:Int) = {
    data.map(r =>
      (r._1, r._2, r._3,
        DenseVector(intToVecProb(r._4,K).toArray),
        DenseVector(intToVecProb(r._5,L).toArray)   )
    )
  }

  def rowPartitionFromRDDPartition(data: RDD[(Int, Array[DenseVector[Double]], DenseVector[Double])]) = {
    data.map(row => (row._1, bzArgMax(row._3))).collect().sortBy(_._1).map(_._2).toList
  }

  def joinLBMBiPartitionToDataCells(data: RDD[(Int, Int, DenseVector[Double])],
                                    rowPartition: List[Int],
                                    colPartition: List[Int])(implicit ss: SparkSession): RDD[(Int, Int, DenseVector[Double], Int, Int)] = {
    data.map(r => (r._1, r._2, r._3, rowPartition(r._1), colPartition(r._2)))
  }

  def joinLBMBiProbBelongingToDataCells(data: RDD[(Int, Int, DenseVector[Double])],
                                        rowProbBelonging: List[List[Double]],
                                        colProbBelonging: List[List[Double]])(implicit ss: SparkSession):
  RDD[(Int, Int, DenseVector[Double], DenseVector[Double], DenseVector[Double])] = {
    val rowProbDvList = ss.sparkContext.broadcast(rowProbBelonging.map(l => DenseVector(l.toArray)))
    val colProbDvList = ss.sparkContext.broadcast(colProbBelonging.map(l => DenseVector(l.toArray)))

    data.map(r => (r._1, r._2, r._3, rowProbDvList.value(r._1), colProbDvList.value(r._2)))
  }

  def joinCLBMBiPartitionToDataCells(data: RDD[(Int, Int, DenseVector[Double])],
                                    rowPartition: List[List[Int]],
                                    colPartition: List[Int])(implicit ss: SparkSession): RDD[(Int, Int, DenseVector[Double], Int, Int)] = {
    data.map(r => {
      val l = colPartition(r._2)
      (r._1, r._2, r._3, rowPartition(l)(r._1), l)
    })
  }
  def getCLBMBiPartitionFromDataCells(data: RDD[(Int, Int, DenseVector[Double], Int, Int)])(implicit ss: SparkSession): (List[List[Int]], List[Int]) = {
    val colPartition = data.map(r => (r._2, r._5)).distinct.collect().sortBy(_._1).map(_._2).toList
    val rowPartition = data.map(r =>
      (r._1, List((r._5, r._4))))
      .reduceByKey(_++_)
      .map(s => (s._1, s._2.distinct.sortBy(_._1).map(_._2)))
      .collect()
      .sortBy(_._1).map(_._2).toList
    (rowPartition.transpose, colPartition)
  }

  def getBiPartitionFromDataCells(data: RDD[(Int, Int, DenseVector[Double], Int, Int)])(implicit ss: SparkSession): (List[Int], List[Int]) = {
    val rowPartition = data.map(r => (r._1, r._4)).distinct.collect().sortBy(_._1).map(_._2).toList
    val colPartition = data.map(r => (r._2, r._5)).distinct.collect().sortBy(_._1).map(_._2).toList
    (rowPartition, colPartition)
  }

  def getBiProbBelongingFromDataCells(data: RDD[(Int, Int, DenseVector[Double], DenseVector[Double], DenseVector[Double])])(implicit ss: SparkSession): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val rowProbBelonging: DenseMatrix[Double] = DenseMatrix(data.map(r => (r._1, r._4)).distinct().collect().sortBy(_._1).map(_._2).toArray:_*)
    val colProbBelonging: DenseMatrix[Double] = DenseMatrix(data.map(r => (r._2, r._5)).distinct().collect().sortBy(_._1).map(_._2).toArray:_*)
    (rowProbBelonging, colProbBelonging)
  }

  def getBiPartitionFromMAPDataCells(data: RDD[(Int, Int, DenseVector[Double], DenseVector[Double], DenseVector[Double])])(implicit ss: SparkSession): (List[Int], List[Int]) = {
    val rowPartition = data.map(r => (r._1, argmax(r._4.toArray.toList))).distinct.collect().sortBy(_._1).map(_._2).toList
    val colPartition = data.map(r => (r._2, argmax(r._5.toArray.toList))).distinct.collect().sortBy(_._1).map(_._2).toList
    (rowPartition, colPartition)
  }

  def getLBMDataPerRowsFromDataPerCells(data: RDD[(Int, Int, DenseVector[Double], Int, Int)]) = {
    data.map(r =>
      (r._1, (Array(r._3), r._4)))
      .reduceByKey((a,b) => (a._1 ++ b._1, a._2))
      .map(r => (r._1, r._2._1, r._2._2))
  }


  def getCLBMDataPerRowsFromDataPerCells(data: RDD[(Int, Int, DenseVector[Double], Int, Int)]) = {

    data.map(r =>
      (r._1, (Array(r._3), List((r._4, r._5)))))
      .reduceByKey((a,b) => (a._1 ++ b._1, a._2 ++ b._2))
      .map(r => (r._1, r._2._1, r._2._2.sortBy(_._1).map(_._2)))
  }

  def getLBMDataPerColsFromDataPerCells(data: RDD[(Int, Int, DenseVector[Double], Int, Int)]) = {
    data.map(r =>
      (r._2, (Array(r._3), r._5)))
      .reduceByKey((a,b) => (a._1 ++ b._1, a._2))
      .map(r => (r._1, r._2._1, r._2._2))
  }


  def joinLBMRowPartitionToData(data: RDD[(Int, Array[DenseVector[Double]])],
                             rowPartition: List[Int],
                             n:Int)(implicit ss: SparkSession): RDD[(Int, Array[DenseVector[Double]], Int)] = {
      val rowPartitionPerRow: List[(Int,Int)] =
        (0 until n).map(i => (i, rowPartition(i))).toList
      data.join(ss.sparkContext.parallelize(rowPartitionPerRow, 30))
        .map(r => {(r._1, r._2._1, r._2._2)})
    }

  def joinRowPartitionToData(data: RDD[(Int, Array[DenseVector[Double]])],
                             rowPartition: List[List[Int]],
                             n:Int)(implicit ss: SparkSession): RDD[(Int, Array[DenseVector[Double]], List[Int])] = {

    val rowPartitionPerRow: List[(Int, List[Int])] = (0 until n).map(i =>
      (i, rowPartition.indices.map(l => rowPartition(l)(i)).toList)
    ).toList
    data.join(ss.sparkContext.parallelize(rowPartitionPerRow, 30)).map(r => {
      (r._1, r._2._1, r._2._2)
    })
  }

}
