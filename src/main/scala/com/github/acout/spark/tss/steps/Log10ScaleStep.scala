package com.github.unsupervise.spark.tss.steps

import com.github.unsupervise.spark.tss.core.TSS
import com.github.unsupervise.spark.tss.functions.{log10, scale}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions

/**
  * Lazy Row Series Log10 Scaling Step Command
  * @param outColName the output column name where to store the log10 version of input series
  * @param inColName the input column name containing the series to which applying element-wise log10 function
  * @param zeroPrecision the log10 precision, enforcing 0 whenever the value is below it
  */
case class Log10ScaleStep(outColName: String, inColName: String, zeroPrecision: Double = 0.00000000000001) extends Step {
  override def apply(in: TSS): TSS = {
    in.addUDFColumn(outColName, inColName,
      functions.udf(scale(zeroPrecision)
        .andThen(log10(1D))
        .andThen((seq: Seq[Double]) => { Vectors.dense(seq.toArray) })))
  }
}
