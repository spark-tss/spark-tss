package com.github.unsupervise.spark.tss.core

import org.apache.spark.broadcast.Broadcast
import spire.ClassTag

/**
  * Created by Anthony Coutant on 20/05/2019.
  */
final case class PrototypeDistsF[V <: Seq[Int] : ClassTag, VV <: Array[Array[Double]], T : ClassTag](
  interPrototypesCachedDistances: Broadcast[VV],
  initialBuffer: T,
  acc: (T, Double) => T,
  bufferToDist: T => Double
) extends Function2[V, V, Double] {

  override def apply(v1: V, v2: V): Double = {
    @annotation.tailrec
    def go(buffer: T, i: Int): Double = {
      if( i < v1.size ) {
        val intermediaryDistance = interPrototypesCachedDistances.value.apply(v1(i))(v2(i))
        go(acc(buffer, intermediaryDistance), i + 1)
      }
      else bufferToDist(buffer)
    }
    go(initialBuffer, 0)
  }

}

object PrototypeDistsF {

  def sum[VV <: Array[Array[Double]]](interPrototypesCachedDistances: Broadcast[VV]) = {
    PrototypeDistsF[Seq[Int], VV, Double](interPrototypesCachedDistances, 0D, _ + _, d => d)
  }

}

/*
def euclideanF[V <: Seq[Double]](v1: V, v2: V) = {
    @annotation.tailrec
    def go(d: Double, i: Int): Double = {
      if( i < v1.size ) {
        val toPow2 = v1(i) - v2(i)
        go(d + toPow2 * toPow2, i + 1)
      }
      else d
    }
    //sqrt(
    go(0D, 0)//)
  }
 */

/*
/**
    * Hamming distance function between two V objects, V inheriting from Seq[Int]
    * @param v1 the first Int "vector" object
    * @param v2 the second Int "vector" object
    * @tparam V the input vector concrete types
    * @return the hamming distance between <v1> and <v2>
    */
  def hammingF[V <: Seq[Int]](v1: V, v2: V) = {
    @annotation.tailrec
    def go(d: Int, i: Int): Double = {
      if( i < v1.size ) {
        val hammingPoint = if(v1(i) != v2(i)) 1 else 0
        go(d + hammingPoint, i + 1)
      }
      else d
    }
    go(0, 0)
  }
 */
