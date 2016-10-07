/**
 *  Copyright (c) 2016 Intel Corporation 
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package org.trustedanalytics.sparktk.dicom.internal.constructors

import java.awt.image.Raster
import java.io._
import java.util.Iterator
import javax.imageio.stream.ImageInputStream
import javax.imageio.{ ImageIO, ImageReader }

import org.apache.commons.io.IOUtils
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.trustedanalytics.sparktk.dicom.Dicom
import org.trustedanalytics.sparktk.frame.Frame
import org.trustedanalytics.sparktk.frame.internal.rdd.FrameRdd

import org.dcm4che3.imageio.plugins.dcm.{ DicomImageReadParam, DicomImageReader }
import org.dcm4che3.io.DicomInputStream
import org.dcm4che3.tool.dcm2xml.org.trustedanalytics.sparktk.Dcm2Xml

object Import extends Serializable {

  /**
   * Get Pixel Data from Dicom Input Stream represented as Array of Bytes
   * @param byteArray Dicom Input Stream represented as Array of Bytes
   * @return DenseMatrix Pixel Data
   */
  def getPixeldata(byteArray: Array[Byte]): DenseMatrix = {

    val pixeldataInputStream = new DataInputStream(new ByteArrayInputStream(byteArray))
    val pixeldicomInputStream = new DicomInputStream(pixeldataInputStream)

    //create matrix
    val iter: Iterator[ImageReader] = ImageIO.getImageReadersByFormatName("DICOM")
    val readers: DicomImageReader = iter.next.asInstanceOf[DicomImageReader]
    val param: DicomImageReadParam = readers.getDefaultReadParam.asInstanceOf[DicomImageReadParam]
    val iis: ImageInputStream = ImageIO.createImageInputStream(pixeldicomInputStream)
    readers.setInput(iis, true)

    //pixels data raster
    val raster: Raster = readers.readRaster(0, param)

    val w = raster.getWidth
    val h = raster.getHeight

    val data = Array.ofDim[Double](h, w)

    for (i <- 0 until h) {
      for (j <- 0 until w) {
        data(i)(j) = raster.getSample(i, j, 0)
      }
    }
    new DenseMatrix(h, w, data.flatten, isTransposed = true)
  }

  /**
   * Get Metadata Xml from Dicom Input Stream represented as byte array
   * @param byteArray Dicom Input Stream represented as byte array
   * @return String Xml Metadata
   */
  def getMetadataXml(byteArray: Array[Byte]): String = {
    val metadataInputStream = new DataInputStream(new ByteArrayInputStream(byteArray))
    val metadataDicomInputStream = new DicomInputStream(metadataInputStream)

    val dcm2xml = new Dcm2Xml()
    val myOutputStream = new ByteArrayOutputStream()
    dcm2xml.convert(metadataDicomInputStream, myOutputStream)
    myOutputStream.toString()
  }

  /**
   * Creates a dicom object with metadata and pixeldata frames
   *
   * @param path Full path to the DICOM files directory
   * @return Dicom object with MetadataFrame and PixeldataFrame
   */
  def importDcm(sc: SparkContext, path: String): Dicom = {

    val dicomFilesRdd = sc.binaryFiles(path)

    val dcmMetadataPixelArrayRDD = dicomFilesRdd.mapPartitions {

      case iter => for {

        (filePath, fileData) <- iter

        // Open PortableDataStream to retrieve the bytes
        fileInputStream = fileData.open()
        byteArray = IOUtils.toByteArray(fileInputStream)

        //Create the metadata xml
        xml = getMetadataXml(byteArray)
        //Create a dense matrix for pixel array
        dm = getPixeldata(byteArray)
        //Metadata
      } yield (xml, dm)
    }.zipWithIndex()

    dcmMetadataPixelArrayRDD.cache()

    val sqlCtx = new SQLContext(sc)
    import sqlCtx.implicits._

    //create metadata pairrdd
    val metaDataPairRDD: RDD[(Long, String)] = dcmMetadataPixelArrayRDD.map {
      case (metadataPixeldata, id) => (id, metadataPixeldata._1)
    }

    val metadataDF = metaDataPairRDD.toDF("id", "metadata")
    val metadataFrameRdd = FrameRdd.toFrameRdd(metadataDF)
    val metadataFrame = new Frame(metadataFrameRdd, metadataFrameRdd.frameSchema)

    //create image matrix pair rdd
    val imageMatrixPairRDD: RDD[(Long, DenseMatrix)] = dcmMetadataPixelArrayRDD.map {
      case (metadataPixeldata, id) => (id, metadataPixeldata._2)
    }

    val imageDF = imageMatrixPairRDD.toDF("id", "imagematrix")
    val pixeldataFrameRdd = FrameRdd.toFrameRdd(imageDF)
    val pixeldataFrame = new Frame(pixeldataFrameRdd, pixeldataFrameRdd.frameSchema)

    new Dicom(metadataFrame, pixeldataFrame)
  }

}
