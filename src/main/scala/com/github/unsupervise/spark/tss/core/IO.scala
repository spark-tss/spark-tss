/**
  * Code imported from https://github.com/EtienneGof/FunCLBM for integration in Spark-TSS
  * Original Author: Etienne Goffinet
  */

package com.github.unsupervise.spark.tss.core

import java.io.{BufferedWriter, FileWriter}

import breeze.linalg.{*, DenseMatrix}
import com.opencsv.CSVWriter

import scala.collection.JavaConverters._
import scala.io.Source
import scala.util.{Failure, Try}

object IO {

  def readDataSet(path: String): List[List[Double]] = {
    val lines = Source.fromFile(path).getLines.toList.drop(1)
    lines.indices.map(seg => {
      lines(seg).drop(1).dropRight(1).split(";").toList.map(string => string.split(":")(1).toDouble)
    }).toList
  }

  def addPrefix(lls: List[List[String]]): List[List[String]] =
    lls.foldLeft((1, List.empty[List[String]])){
      case ((serial: Int, acc: List[List[String]]), value: List[String]) =>
        (serial + 1, (serial.toString +: value) +: acc)
    }._2.reverse


  def writeMatrixStringToCsv(fileName: String, Matrix: DenseMatrix[String], append: Boolean = false): Unit = {
    val rows : List[List[String]] = Matrix(*, ::).map(dv => dv.toArray.toList).toArray.toList
    writeCsvFile(fileName, addPrefix(rows), append=append)
  }

  def writeMatrixDoubleToCsv(fileName: String, Matrix: DenseMatrix[Double], withHeader:Boolean=true): Unit = {
    val header: List[String] = List("id") ++ (0 until Matrix.cols).map(_.toString).toList
    val rows : List[List[String]] = Matrix(*, ::).map(dv => dv.toArray.map(_.toString).toList).toArray.toList
    if(withHeader){
      writeCsvFile(fileName, addPrefix(rows), header)
    } else {
      writeCsvFile(fileName, addPrefix(rows))
    }
  }

  def writeMatrixIntToCsv(fileName: String, Matrix: DenseMatrix[Int], withHeader:Boolean=true): Unit = {
    val header: List[String] = List("id") ++ (0 until Matrix.cols).map(_.toString).toList
    val rows : List[List[String]] = Matrix(*, ::).map(dv => dv.toArray.map(_.toString).toList).toArray.toList
    if(withHeader){
      writeCsvFile(fileName, addPrefix(rows), header)
    } else {
      writeCsvFile(fileName, addPrefix(rows))
    }
  }

  def writeCsvFile(fileName: String,
                   rows: List[List[String]],
                   header: List[String] = List.empty[String],
                   append:Boolean=false
                  ): Try[Unit] =
  {
    val content = if(header.isEmpty){rows} else {header +: rows}
    Try(new CSVWriter(new BufferedWriter(new FileWriter(fileName, append)))).flatMap((csvWriter: CSVWriter) =>
      Try{
        csvWriter.writeAll(
          content.map(_.toArray).asJava
        )
        csvWriter.close()
      } match {
        case f @ Failure(_) =>
          // Always return the original failure.  In production code we might
          // define a new exception which wraps both exceptions in the case
          // they both fail, but that is omitted here.
          Try(csvWriter.close()).recoverWith{
            case _ => f
          }
        case success =>
          success
      }
    )
  }
}