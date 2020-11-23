/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.funlbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import breeze.linalg.{DenseVector, argmax}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.annotation.tailrec


object ModelSelection {

  def  gridSearch(data: RDD[(Int, Array[DenseVector[Double]])],
                  EMMethod: String= "SEMGibbs",
                  rangeRow:List[Int],
                  rangeCol:List[Int],
                  verbose: Boolean = false,
                  nConcurrent:Int=3,
                  nTryMaxPerConcurrent:Int=10,
                  updateLoadingsStrategy: Int=3)(implicit ss: SparkSession): Map[String,Product] = {

    val gridRange: List[(Int, Int)] = (rangeRow cross rangeCol).toList
    val allRes = gridRange.map(KL => {
      if(verbose) {println()
        println(">>>>> LBM Grid Search try: (K:"+KL._1.toString+", L:"+KL._2.toString+")")}
      val latentBlock = new FunLatentBlock()
      latentBlock.setK(KL._1).setL(KL._2).setMaxIterations(7).setMaxBurninIterations(7).setUpdateLoadingStrategy(updateLoadingsStrategy)
      val res = latentBlock.run(data,EMMethod, nConcurrent=nConcurrent, nTryMaxPerConcurrent = nTryMaxPerConcurrent, verbose=verbose)
      println(res("ColPartition").asInstanceOf[List[Int]])
      res
    })

    val allIcl: DenseVector[Double] = DenseVector(allRes.map(_("ICL")
      .asInstanceOf[List[Double]].last).toArray)

    if(verbose) {println()
      (gridRange zip allIcl.toArray.toList).foreach(icl => println("Icl :"+icl))}

    allRes(argmax(allIcl))
  }

  class Factorial2 {
    def factorial(n: Int): Int = {
      @tailrec def factorialAcc(acc: Int, n: Int): Int = {
        if (n <= 1) acc
        else factorialAcc(n * acc, n - 1)
      }
      factorialAcc(1, n)
    }
  }

  def greedySearch(data: RDD[(Int, Array[DenseVector[Double]])],
                 EMMethod: String= "SEMGibbs",
                 maxK: Int,
                 maxL: Int,
                 verbose: Boolean = false,
                 nConcurrentEachTest:Int=3,
                 nTryMaxPerConcurrent:Int=10)(implicit ss: SparkSession): Map[String,Product] = {

    val prevLB = new FunLatentBlock().setK(1).setL(1)
    prevLB.setMaxIterations(7).setMaxBurninIterations(7).setUpdateLoadingStrategy("never")
    var prevRes = prevLB.run(data,EMMethod, nConcurrent=nConcurrentEachTest, nTryMaxPerConcurrent = nTryMaxPerConcurrent, verbose=verbose)
    val prevModel = prevRes("Model").asInstanceOf[FunLatentBlockModel]
    var prevIcl = prevRes("ICL").asInstanceOf[List[Double]].last

    val tmpLB = new FunLatentBlock().setMaxIterations(5).setMaxBurninIterations(5).setUpdateLoadingStrategy(4)

    if(verbose){println("Initial Icl: "+prevIcl.toString)}

    @tailrec def go(prevRes: Map[String,Product], prevModel: FunLatentBlockModel, prevIcl: Double): Map[String,Product] = {

      val resCandidateK = tmpLB.setK(prevModel.K+1).setL(prevModel.L).run(data,EMMethod, nConcurrent=nConcurrentEachTest, nTryMaxPerConcurrent = nTryMaxPerConcurrent, verbose=verbose)
      val resCandidateL = tmpLB.setK(prevModel.K).setL(prevModel.L+1).run(data,EMMethod, nConcurrent=nConcurrentEachTest, nTryMaxPerConcurrent = nTryMaxPerConcurrent, verbose=verbose)
      val candidateModelK = resCandidateK("Model").asInstanceOf[FunLatentBlockModel]
      val candidateModelL= resCandidateL("Model").asInstanceOf[FunLatentBlockModel]
      val candidateIclK = resCandidateK("ICL").asInstanceOf[List[Double]].last
      val candidateIclL = resCandidateL("ICL").asInstanceOf[List[Double]].last
      if(verbose){println("Candidate K+1: ("+(prevModel.K+1).toString+","+prevModel.L.toString+")'s Icl: "+candidateIclK.toString)}
      if(verbose){println("Candidate L+1: ("+prevModel.K.toString+","+(prevModel.L+1).toString+")'s Icl: "+candidateIclL.toString)}
      val (bestRes, bestModel, bestIcl) = if(candidateIclK > candidateIclL){
        (resCandidateK, candidateModelK,candidateIclK)} else
        (resCandidateL, candidateModelL,candidateIclL)
      if(prevLB.getK <= maxK && prevLB.getL <= maxL && prevIcl < bestIcl){
        go(bestRes, bestModel, bestIcl)
      } else {
        prevRes
      }
    }
    go(prevRes, prevModel, prevIcl)
  }
}
