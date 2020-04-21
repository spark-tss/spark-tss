package com.github.acout.spark.tss.steps

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
    * Build a Pipeline by appending another Step to <this>
    * @param nextStep the Step to chain after <this> Step
    * @return a Pipeline beginning with <this> and then ending with <nextStep>
    */
  def +(nextStep: Step) = Pipeline(Seq(this, nextStep))
}
