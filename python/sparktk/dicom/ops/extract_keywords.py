def extract_keywords(self, keywords):
    """

    Extract value for each keyword from column holding xml string

    Ex: keywords -> ["PatientID"]

    Parameters
    ----------

    :param keywords: (str or list(str)) List of keywords from metadata xml string


    Examples
    --------

        <skip>
        >>> dicom_path = "../datasets/dicom_uncompressed"

        >>> dicom = tc.dicom.import_dcm(dicom_path)

        >>> dicom.metadata.inspect(truncate=30)
        [#]  id  metadata
        =======================================
        [0]   0  <?xml version="1.0" encodin...
        [1]   1  <?xml version="1.0" encodin...
        [2]   2  <?xml version="1.0" encodin...

        #Extract values for given keywords and add as new columns in metadata frame
        >>> dicom.extract_keywords(["SOPInstanceUID", "Manufacturer", "StudyDate"])

        >>> dicom.metadata.inspect(truncate=20)
        [#]  id  metadata              SOPInstanceUID        Manufacturer  StudyDate
        ============================================================================
        [0]   0  <?xml version="1....  1.3.12.2.1107.5.2...  SIEMENS       20040305
        [1]   1  <?xml version="1....  1.3.12.2.1107.5.2...  SIEMENS       20040305
        [2]   2  <?xml version="1....  1.3.12.2.1107.5.2...  SIEMENS       20040305
        </skip>

    """

    if isinstance(keywords, basestring):
        keywords = [keywords]

    if self._metadata._is_scala:
        def f(scala_dicom):
            scala_dicom.extractKeywords(self._tc.jutils.convert.to_scala_vector_string(keywords))
        results = self._call_scala(f)
        return results

    # metadata is python frame, run below udf
    import xml.etree.ElementTree as ET

    def extractor(dkeywords):
        def _extractor(row):
            root = ET.fromstring(row["metadata"])
            values=[None]*len(dkeywords)
            for attribute in root.findall('DicomAttribute'):
                dkeyword = attribute.get('keyword')
                if dkeyword in dkeywords:
                    if attribute.find('Value') is not None:
                        values[dkeywords.index(dkeyword)]=attribute.find('Value').text
            return values
        return _extractor

    cols_type = [(key, str) for key in keywords]
    self._metadata.add_columns(extractor(keywords), cols_type)



