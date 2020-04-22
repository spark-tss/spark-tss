package com.github.unsupervise.spark.tss.steps

import java.io.File

import com.github.unsupervise.spark.tss.core.TSS

/**
  * Lazy SOM Step Command.
  * The class is designed to perform several clustering computations at once, thus has only Array parameters.
  * @param outColNames the Array of hard prediction output column names for each clustering to perform
  * @param inColName the input column name to use for all clusterings (single input, multiple outputs fashion)
  * @param ks the Array of cluster grid (height, width) pairs to use for each clustering to perform
  * @param maxIterations the number of maximal iterations to perform for each clustering
  * @param centerFilePaths the Array of file paths where to store the final clustering centers
  */
case class SOMStep(outColNames: Array[String], inColName: String, ks: Array[(Int, Int)], maxIterations: Int, centerFilePaths: Array[String]) extends Step {

  override def apply(in: TSS): TSS = {
    ks.indices.foldLeft(in){
      case (tss, i) => tss.addSOM(outColNames(i), inColName, ks(i)._1, ks(i)._2, maxIterations, Some(new File(centerFilePaths(i))))
    }
  }

}
