/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.lbmtools

import breeze.linalg.{DenseMatrix, DenseVector, max, sum}
import org.apache.spark.ml.linalg.{Matrices, Matrix}

import scala.util.{Success, Try}

object Tools extends java.io.Serializable {

  def prettyPrint(sizePerBlock: Map[(Int,Int), Int]) = {

    val keys = sizePerBlock.keys
    val L = keys.map(_._1).max + 1
    val K = keys.map(_._2).max + 1
    val mat = DenseMatrix.tabulate[String](L,K){
      case (i, j) => {
        if(sizePerBlock.contains((i,j))){
          sizePerBlock(i,j).toString
        } else {"-"}
      }
    }

    println(mat.t)

  }


  def nestedMap[A, B, C](listA: List[List[A]],listB: List[List[B]])(f: (A,B) => C): List[List[C]] = {
    require(listA.length == listB.length)
    listA.indices.foreach(i => require(listA(i).length== listB(i).length))
    listA.indices.map(i => listA(i).indices.map(j => f(listA(i)(j), listB(i)(j))).toList).toList
  }

  def getPartitionFromSize(size: List[Int]) = {
    size.indices.map(idx => List.fill(size(idx))(idx)).reduce(_ ++ _)
  }

  def aggregateListListDV(a: List[List[DenseVector[Double]]],
                          b: List[List[DenseVector[Double]]]): List[List[DenseVector[Double]]] = {
    a.indices.map(l => {
      a(l).indices.map(k => {
        a(l)(k) + b(l)(k)
      }).toList
    }).toList
  }

  def getProportions(l: List[Int]): List[Double] = {
    val countCols = l.groupBy(identity).mapValues(_.size)
    countCols.map(c => c._2 / countCols.values.sum.toDouble).toList
  }

  def argmax(l: List[Double]): Int ={
    l.view.zipWithIndex.maxBy(_._1)._2
  }

  def getTime[R](block: => R): Double = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    (t1 - t0)/ 1e9
  }

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0)/ 1e9 + "s")
    result
  }

  def printTime(t0:Long, stepName: String, verbose:Boolean = true): Long={
    if(verbose){
      println(stepName.concat(" step duration: ").concat(((System.nanoTime - t0)/1e9D ).toString))
    }
    System.nanoTime
  }

  implicit class Crossable[X](xs: Traversable[X]) {
    def cross[Y](ys: Traversable[Y]) : Traversable[(X,Y)] = for { x <- xs; y <- ys } yield (x, y)
  }

  def denseMatrixToMatrix(A: DenseMatrix[Double]): Matrix = {
    Matrices.dense(A.rows, A.cols, A.toArray)
  }

  def checkPartitionEqual(partitionA : List[Int], partitionB: List[Int]): Boolean = {
    require(partitionA.length == partitionB.length)
    val uniqueA = partitionA.distinct.zipWithIndex
    val uniqueB = partitionB.distinct.zipWithIndex
    val dictA = (for ((k, v) <- uniqueA) yield (v, k)).toMap
    val dictB = (for ((k, v) <- uniqueB) yield (v, k)).toMap
    val reOrderedA = partitionA.map(e => dictA(e))
    val reOrderedB = partitionB.map(e => dictB(e))

    sum(DenseVector(reOrderedA.toArray)-DenseVector(reOrderedB.toArray))==0
  }

  def matrixToDenseMatrix(A: Matrix): DenseMatrix[Double] = {
    val p = A.numCols
    val n = A.numRows
    DenseMatrix(A.toArray).reshape(n,p)
  }

  def isInteger(x: String) : Boolean = {
    val y = Try(x.toInt)
    y match {
      case Success(_) => true
      case _ => false
    }
  }

  def inverseIndexedList(data: List[(Int, Array[DenseVector[Double]])]): List[(Int, Array[DenseVector[Double]])] = {
    val p = data.take(1).head._2.length
    (0 until p).map(j => {
      (j, data.map(row => row._2(j)).toArray)
    }).toList
  }

  def getEntireRowPartition(rowPartition: List[List[Int]]): Array[Int] = {
    val n = rowPartition.head.length
    val L = rowPartition.length
    val rowMultiPartition: List[List[Int]] = (0 until n).map(i => (0 until L).map(l => rowPartition(l)(i)).toList).toList

    val mapMultiPartitionToRowCluster = rowMultiPartition.distinct.zipWithIndex.toMap
    rowMultiPartition.map(mapMultiPartitionToRowCluster(_)).toArray
  }

  def remove[T](list: List[T], idx: Int):List[T] = list.patch(idx, Nil, 1)

  def insert[T](list: List[T], i: Int, values: T*): List[T] = {
    val (front, back) = list.splitAt(i)
    front ++ values ++ back
  }

  def roundMat(m: DenseMatrix[Double], digits:Int=0): DenseMatrix[Double] = {
    m.map(round(_,digits))
  }

  def roundDv(m: DenseVector[Double], digits:Int=0): DenseVector[Double] = {
    m.map(round(_,digits))
  }


  def round(x: Double, digits:Int=0): Double = {
    val factor: Double = Math.pow(10,digits)
    Math.round(x*factor)/factor
  }

  def allEqual[T](x: List[T], y:List[T]): Boolean = {
    require(x.length == y.length)
    val listBool = x.indices.map(i => {x(i)==y(i)})
    listBool.forall(identity)
  }

  def getCondBlockPartition(rowPartition: List[List[Int]], colPartition: List[Int]): List[(Int, Int)] = {
    val blockPartitionMat = colPartition.par.map(l => {
      DenseMatrix.tabulate[(Int, Int)](rowPartition.head.length, 1) {
        (i, _) => (rowPartition(l)(i), l)
      }}).reduce(DenseMatrix.horzcat(_,_))
    blockPartitionMat.t.toArray.toList
  }

  def getBlockPartition(rowPartition: List[List[Int]], colPartition: List[Int]): Array[Int] = {
    val n = rowPartition.head.length
    val p = colPartition.length
    val blockBiPartition: List[(Int, Int)] = DenseMatrix.tabulate[(Int, Int)](n,p)(
      (i, j) => (rowPartition(colPartition(j))(i), colPartition(j))
    ).toArray.toList
    val mapBlockBiIndexToBlockNum = blockBiPartition.distinct.zipWithIndex.toMap
    blockBiPartition.map(mapBlockBiIndexToBlockNum(_)).toArray
  }


  def updateColPartition(formerColPartition: List[Int],
                         colToUpdate: Int,
                         newColPartition: List[Int]): List[Int]={
    var newGlobalColPartition = formerColPartition
    val otherThanlMap:Map[Int,Int] = formerColPartition.filter(_!=colToUpdate).distinct.sorted.zipWithIndex.toMap
    val L = max(formerColPartition)

    var iterNewColPartition = 0
    for( j <- newGlobalColPartition.indices){
      if(formerColPartition(j)==colToUpdate){
        newGlobalColPartition = newGlobalColPartition.updated(j,newColPartition(iterNewColPartition)+L)
        iterNewColPartition +=1
      } else {
        newGlobalColPartition = newGlobalColPartition.updated(j,otherThanlMap(formerColPartition(j)))
      }
    }
    newGlobalColPartition
  }

  def generateCombinationWithReplacement(maxK: Int, L: Int): List[List[Int]] ={
    List.fill(L)((1 to maxK).toList).flatten.combinations(L).toList
  }
}
