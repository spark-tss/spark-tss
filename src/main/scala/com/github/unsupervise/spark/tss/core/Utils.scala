package com.github.unsupervise.spark.tss.core

/**
  * Created by Anthony Coutant on 20/06/2019.
  */
object Utils {
  /**
   * Sums together things in log space.
   * @return log(exp(a) + exp(b))
   */
  def logSumExp(a: Double, b: Double) = {
    if (a == Double.NegativeInfinity) b
    else if (b == Double.NegativeInfinity) a
    else if (a < b) b + Math.log(1 + Math.exp(a - b))
    else a + Math.log(1 + Math.exp(b - a))
  }
  /**
    * Sums together things in log space.
    * @return log(\sum exp(a_i))
    */
  def logSumExp(a: Seq[Double]):Double = {
    a.length match {
      case 0 => Double.NegativeInfinity
      case 1 => a(0)
      case 2 => logSumExp(a(0),a(1))
      case _ =>
        val m = a reduceLeft(Math.max)
        if (m.isInfinite) m
        else {
          var i = 0
          var accum = 0.0
          while(i < a.length) {
            accum +=  Math.exp(a(i) - m)
            i += 1
          }
          m +  Math.log(accum)
        }
    }
  }
  def sample(probabilities: List[Double]): Int = {
    val dist = probabilities.indices zip probabilities
    val threshold = scala.util.Random.nextDouble
    val iterator = dist.iterator
    var accumulator = 0.0
    while (iterator.hasNext) {
      val (cluster, clusterProb) = iterator.next
      accumulator += clusterProb
      if (accumulator >= threshold)
        return cluster
    }
    sys.error("Error")
  }

}
