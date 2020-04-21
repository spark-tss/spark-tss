package com.github.acout.spark.tss.steps

import com.github.acout.spark.tss.bench.TSSBench
import com.github.acout.spark.tss.core.TSS
import org.apache.spark.sql.SparkSession

/**
  * A Lazy Computation Unit which can be concatenate with each other to form a Pipeline
  */
trait Step {
  /**
    * Actually apply the Step to a given TSS
    * @param in the input TSS
    * @return a TSS being the result of the Step computation
    */
  def apply(in: TSS): TSS

  /**
    * Benchmark the partitions and running time of the Step for a given TSS stored on disk
    * @param inPath the path where the input TSS to compute the Step on is stored
    * @param outPath the path where the output TSS must be stored, enforcing the computation of the Step.
    * @param overwrite whether to overwrite the output file
    * @param ss
    */
  def bench(inPath: String, outPath: String, overwrite: Boolean = true)(implicit ss: SparkSession) = {
    val tss = TSS.load(inPath)
    TSSBench.timePrint(apply(tss).save(outPath, overwrite))
  }

  /**
    * Build a Pipeline by appending another Step to <this>
    * @param nextStep the Step to chain after <this> Step
    * @return a Pipeline beginning with <this> and then ending with <nextStep>
    */
  def +(nextStep: Step) = Pipeline(Seq(this, nextStep))
}
