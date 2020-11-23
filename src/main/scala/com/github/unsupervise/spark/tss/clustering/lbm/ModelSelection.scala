/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.clustering.lbm

import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools._
import breeze.linalg.{DenseVector, argmax}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object ModelSelection {

  def gridSearch(data: RDD[(Int, Array[DenseVector[Double]])],
                 EMMethod: String= "SEMGibbs",
                 rangeRow:List[Int],
                 rangeCol:List[Int],
                 verbose: Boolean = false,
                 nConcurrentEachTest:Int=1,
                 nTryMaxPerConcurrent:Int=30)(implicit ss: SparkSession): Map[String,Product] = {

    var latentBlock = new LatentBlock()
    val gridRange: List[(Int, Int)] = (rangeRow cross rangeCol).toList
    val allRes = gridRange.map(KL => {
      if(verbose) {println()
        println(">>>>> LBM Grid Search try: (K:"+KL._1.toString+", L:"+KL._2.toString+")")}
      latentBlock.setK(KL._1).setL(KL._2)
      latentBlock.run(data,
        EMMethod,
        nConcurrent=nConcurrentEachTest,
        nTryMaxPerConcurrent = nTryMaxPerConcurrent,
        verbose=verbose)
    })

    val Loglikelihoods: DenseVector[Double] = DenseVector(allRes.map(_("LogLikelihood")
      .asInstanceOf[List[Double]].last).toArray)
    val ICLs: DenseVector[Double] = DenseVector(allRes.map(_("ICL")
      .asInstanceOf[List[Double]].last).toArray)

    if(verbose) {
      println()
      gridRange.indices.foreach(i => {
        println("("+gridRange(i)._1+", "+ gridRange(i)._2+"), Loglikelihood: ", Loglikelihoods(i)+", ICL: "+ICLs(i))
      })
    }

    allRes(argmax(ICLs))
  }


}
