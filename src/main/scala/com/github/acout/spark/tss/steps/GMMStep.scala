package com.github.acout.spark.tss.steps

import java.io.File

import com.github.acout.spark.tss.core.TSS

/**
  * Lazy GMM Step Command.
  * The class is designed to perform several clustering computations at once, thus has only Array parameters.
  * @param outHardColNames the Array of hard prediction output column names for each clustering to perform
  * @param outSoftColNames the Array of soft prediction output column names for each clustering to perform
  * @param outLLColNames the Array of per sample log likelihood output column names for each clustering to perform
  * @param inColName the input column name to use for all clusterings (single input, multiple outputs fashion)
  * @param ks the Array of cluster numbers to use for each clustering to perform
  * @param maxIterations the number of maximal iterations to perform for each clustering
  * @param kMeansInit whether to initialize GMM with KMeans in each clustering
  * @param runsNb the number of runs among which taking the best for each clustering
  * @param centerFilePaths the Array of file paths where to store the final clustering centers
  */
case class GMMStep(outHardColNames: Array[String], outSoftColNames: Array[String], outLLColNames: Array[String], inColName: String, ks: Array[Int], maxIterations: Int, kMeansInit: Boolean, runsNb: Int, centerFilePaths: Array[String], fullModelPaths: Array[String]) extends Step {

  override def apply(in: TSS): TSS = {
    ks.indices.foldLeft(in){
      case (tss, i) => {
        println(outHardColNames(i) + " " + ks(i))
        try{
          //println(tss.series.rdd.mapPartitions(x => Iterator(x.size)).collect().sorted.mkString(", "))
          //println("dump partition pre")
          //TSSBench.dumpPartition(tss)
          val tsss = tss.addGMM(outHardColNames(i), outSoftColNames(i), outLLColNames(i), inColName, ks(i), maxIterations, kMeansInit, runsNb, Some(new File(centerFilePaths(i))), Some(fullModelPaths(i)))//.repartition(128)
          //println("dump partition post")
          //TSSBench.dumpPartition(tsss
          /*println("Partitioner: " + tsss.series.rdd.partitioner)
          println("Num Partitions: " + tsss.series.rdd.partitions.size)
          println(tsss.series.rdd.mapPartitions(x => Iterator(x.size)).collect().sorted.mkString(", "))*/
          tsss
        }catch{
          case e => {
            println(e)
            tss
          }
        }
      }
    }
  }

}
