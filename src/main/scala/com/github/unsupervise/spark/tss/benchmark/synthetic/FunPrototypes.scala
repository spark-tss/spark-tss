/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  * Adapted by: Anthony Coutant
  */

package com.github.unsupervise.spark.tss.benchmark.synthetic

import com.github.unsupervise.spark.tss.clustering.lbmtools.Tools.allEqual
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.numerics.sin
import breeze.stats.distributions.{Gaussian, MultivariateGaussian}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.math._
import scala.util.Random

//TODO: Propose prototype generators
object FunPrototypes  {

  // Sin prototype
  def defaultSin = (x: List[Double]) => {
    x.map(t => 1+0.5*sin(4*scala.math.Pi*t))
  }

  // Sigmoid prototype
  def defaultSigmoid = (x: List[Double]) => {
    val center = Gaussian(0.6,0.02).draw()
    val slope = 20D
    val maxVal =  Gaussian(1,0.02).draw()
    x.map(t => maxVal/(1+exp(-slope*(t-center))))
  }

  // Rectangular prototype
  def defaultRect = (x: List[Double]) => {
    val start = 0.3
    val duration = 0.3
    x.map({
      case t if t <`start` || t>=(`start`+`duration`) => 0D
      case _ => 1D
    })
  }

  // Morlet prototype
  def defaultMorlet = (x: List[Double]) => {
    val center = 0.5
    x.map(t => {
      val u = (t-center)*10
      exp(-0.5*pow(u,2))*cos(5*u)
    })
  }

  // Gaussian prototype
  def defaultGaussian = (x: List[Double]) => {
    val center = 0.5
    val sd = 0.1
    val G = Gaussian(center, sd)
    x.map(t => 1.5*G.pdf(t)/G.pdf(center))
  }

  // Double Gaussian prototype
  def defaultDoubleGaussian = (x: List[Double]) => {
    val center1 = 0.2
    val center2 = 0.7
    val sd = 0.1
    val G1 = Gaussian(center1, sd)
    val G2 = Gaussian(center2, sd)

    x.map(t => G1.pdf(t)/G1.pdf(center1)+ G2.pdf(t)/G2.pdf(center2))
  }

  // y-shifted Gaussian prototype
  def defaultYShiftedGaussian = (x: List[Double]) => {
    x.map(t => 0.3+ 0.3*sin(2*scala.math.Pi*t))
  }

  // sin by rectangular prototype
  def defaultSinRect = (x: List[Double]) => {
    val start = 0.3
    val duration = 0.3
    x.map({
      case t if t <`start` || t>=(`start`+`duration`) => -1 + 0.5* sin(2*Pi*t)
      case t => 2+  0.5*sin(2*Pi*t)
    })
  }

}
