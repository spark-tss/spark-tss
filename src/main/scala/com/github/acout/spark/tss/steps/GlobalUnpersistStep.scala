package com.github.acout.spark.tss.steps

import com.github.acout.spark.tss.core.TSS
import org.apache.spark.rdd.RDD

/**
  * Lazy Global Pipeline RDDs & DF Unpersist Step Command.
  * This Step destroys all cached and persisted RDDs and DFs even when references to them have been lost.
  * @param filter an optional filter to restrict the destruction to subset of RDD and DFs.
  */
case class GlobalUnpersistStep(filter: Function1[(Int, RDD[_]), Boolean] = _ => true) extends Step {
  /**
    * Actually apply the Step to a given TSS
    *
    * @param in the input TSS
    * @return a TSS being the result of the Step computation
    */
  override def apply(in: TSS): TSS = {
    in.series.sparkSession.sparkContext.getPersistentRDDs.filter(filter).foreach(_._2.unpersist())
    System.gc()
    in
  }
}
