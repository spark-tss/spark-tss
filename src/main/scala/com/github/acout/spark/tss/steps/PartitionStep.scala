package com.github.acout.spark.tss.steps

import com.github.acout.spark.tss.core.TSS
import org.apache.spark.sql.Column

/**
  * Lazy Partition Step Command.
  * @param partitionsNumber the number of partitions in output
  * @param cols the optional list of partition columns
  */
case class PartitionStep(partitionsNumber: Int, cols: Option[Seq[Column]] = None) extends Step {

  /**
    * Actually apply the Step to a given TSS
    *
    * @param in the input TSS
    * @return a TSS being the result of the Step computation
    */
  override def apply(in: TSS): TSS = {
    cols match {
      case Some(cs) => in.repartition(partitionsNumber, cs: _*)
      case None => in.repartition(partitionsNumber)
    }
  }

}
