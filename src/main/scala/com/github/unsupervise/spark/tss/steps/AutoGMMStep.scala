package com.github.unsupervise.spark.tss.steps

import java.io.File

import com.github.unsupervise.spark.tss.core.TSS

/**
  * Lazy Automatic ML GMM Step Command
  * @param outHardColName the hard prediction output column name
  * @param outSoftColName the soft prediction output column name
  * @param outLLColName the per sample log likelihood output column name
  * @param inColName the input column name to use for clustering
  * @param kBounds the bottom and top bounds defining a closed interval for the admissible number of clusters k into which the best value will be automatically picked
  * @param initIter the initial number of clustering to performs to initialize the k space to optimize
  * @param optimIter the after initialization number of AutoML iterations
  * @param maxIter the number of maximal iterations to perform for each clustering
  * @param kMeansInit whether to initialize GMM with KMeans
  * @param runsNb the number of runs among which taking the best for each clustering
  * @param centerFilePath the file path where to store the final clustering centers
  */
case class AutoGMMStep(outHardColName: String, outSoftColName: String, outLLColName: String, inColName: String, kBounds: (Int, Int), initIter: Int, optimIter: Int,  maxIter: Int, kMeansInit: Boolean, runsNb: Int, centerFilePath: String, fullModelOutPath: String) extends Step {
  override def apply(in: TSS): TSS = {
    in.addAutoGMM(outHardColName, outSoftColName, outLLColName, inColName, kBounds, initIter, optimIter, maxIter, kMeansInit, runsNb, Some(new File(centerFilePath)), Some(fullModelOutPath))(in.series.sparkSession)
  }
}
