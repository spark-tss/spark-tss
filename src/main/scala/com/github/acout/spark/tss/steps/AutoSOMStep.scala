package com.github.acout.spark.tss.steps

import java.io.File

import com.github.acout.spark.tss.core.TSS

/**
  * Lazy Automatic ML SOM Step Command
  * @param outHardColName the hard prediction output column name
  * @param inColName the input column name to use for clustering
  * @param heightBounds the bottom and top bounds defining a closed interval for the admissible number of clusters grid height into which the best value will be automatically picked
  * @param widthBounds the bottom and top bounds defining a closed interval for the admissible number of clusters grid width into which the best value will be automatically picked
  * @param initIter the initial number of clustering to performs to initialize the clusters grid space to optimize
  * @param optimIter the after initialization number of AutoML iterations
  * @param maxIter the number of maximal iterations to perform for each clustering
  * @param runsNb the number of runs among which taking the best for each clustering
  * @param centerFilePath the file path where to store the final clustering centers
  */
case class AutoSOMStep(outHardColName: String, inColName: String, heightBounds: (Int, Int), widthBounds: (Int, Int), initIter: Int, optimIter: Int, maxIter: Int, runsNb: Int, centerFilePath: String) extends Step {
  override def apply(in: TSS): TSS = {
    in.addAutoSOM(outHardColName, inColName, heightBounds, widthBounds, initIter, optimIter, maxIter, runsNb, Some(new File(centerFilePath)))(in.series.sparkSession)
  }
}
