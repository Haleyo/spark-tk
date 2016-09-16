package org.trustedanalytics.sparktk.dicom.internal.ops

import org.apache.spark.sql.Row
import org.trustedanalytics.sparktk.dicom.internal.{ BaseDicom, DicomTransform, DicomState }
import org.trustedanalytics.sparktk.frame.internal.rdd.RowWrapperFunctions
import org.trustedanalytics.sparktk.frame._
import org.trustedanalytics.sparktk.frame.internal._

import scala.xml.NodeSeq

trait ExtractKeywordsTransform extends BaseDicom {

  /**
   * Extracts the value for each keyword from column holding xml string
   *
   * @param keywords keywords to extract from column holding xml string
   */
  def extractKeywords(keywords: Seq[String]) = {
    execute(ExtractKeywords(keywords))
  }
}

case class ExtractKeywords(keywords: Seq[String]) extends DicomTransform {

  override def work(state: DicomState): DicomState = {
    ExtractKeywords.extractKeywordsImpl(state.metadata, keywords)
    state
  }
}

object ExtractKeywords extends Serializable {

  private implicit def rowWrapperToRowWrapperFunctions(rowWrapper: RowWrapper): RowWrapperFunctions = {
    new RowWrapperFunctions(rowWrapper)
  }

  //Get value if keyword exists else return null
  def getKeywordValue(nodeSeqOfDicomAttribute: NodeSeq)(keyword: String): String = {
    val resultNodeSeq = nodeSeqOfDicomAttribute.filter {
      da => (da \ "@keyword").text == keyword
    }
    if (resultNodeSeq.nonEmpty)
      resultNodeSeq.head.text
    else
      null
  }

  /**
   * Custom RowWrapper to apply on each row
   *
   * @param keywords keywords to add as columns
   * @return Row
   */
  private def customDicomAttributeRowWrapper(keywords: Seq[String]) = {
    val rowMapper: RowWrapper => Row = row => {
      val columnName = "metadata" //This should be name of the column holding xml string as value in a frame
      val nodeName = "DicomAttribute" //This should be node name in xml string

      //Creates NodeSeq of DicomAttribute
      val nodeSeqOfDicomAttribute = row.valueAsNodeSeq(columnName, nodeName)

      //Filter each DicomAttribute node with given keyword and extract value
      val nodeValues = keywords.map(getKeywordValue(nodeSeqOfDicomAttribute))

      //Creates a Row from given Sequence of node values
      Row.fromSeq(nodeValues)
    }
    rowMapper
  }

  /**
   * Extracts the value for each keyword from column holding xml string
   *
   * @param metadataFrame metadata frame with column holding xml string
   * @param keywords keywords to extract from column holding xml string
   */
  def extractKeywordsImpl(metadataFrame: Frame, keywords: Seq[String]) = {
    val newColumns = for (keyword <- keywords) yield Column(keyword, DataTypes.string)
    metadataFrame.addColumns(customDicomAttributeRowWrapper(keywords), newColumns)
  }

}
