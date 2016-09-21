def drop_rows_by_tags(self, tags_values_dict):
    """
    Drop the rows based on dictionary of {"tag":"value"} from column holding xml string

    Ex: tags_values_dict -> {"00080018":"1.3.12.2.1107.5.2.5.11090.5.0.5823667428974336", "00080070":"SIEMENS", "00080020":"20040305"}

    Parameters
    ----------

    :param tags_values_dict: (dict(str, str)) dictionary of tags and values from xml string in metadata


    Examples
    --------

        >>> dicom_path = "../datasets/dicom_uncompressed"

        >>> dicom = tc.dicom.import_dcm(dicom_path)

        <skip>
        >>> dicom.metadata.inspect(truncate=30)
        [#]  id  metadata
        =======================================
        [0]   0  <?xml version="1.0" encodin...
        [1]   1  <?xml version="1.0" encodin...
        [2]   2  <?xml version="1.0" encodin...
        </skip>

        >>> tags_values_dict = {"00080018":"1.3.12.2.1107.5.2.5.11090.5.0.5823667428974336", "00080070":"SIEMENS", "00080020":"20040305"}
        >>> dicom.drop_rows_by_tags(tags_values_dict)
        >>> dicom.metadata.count()
        2

        <skip>
        #After drop_rows
        >>> dicom.metadata.inspect(truncate=30)
        [#]  id  metadata
        =======================================
        [0]   1  <?xml version="1.0" encodin...
        [1]   2  <?xml version="1.0" encodin...

        >>> dicom.pixeldata.inspect(truncate=30)
        [#]  id  imagematrix
        =========================================
        [1]   1  [[  0.   1.   0. ...,   0.   0.   1.]
        [  1.   9.  10. ...,   2.   4.   6.]
        [  0.  12.  11. ...,   4.   4.   7.]
        ...,
        [  0.   4.   2. ...,   3.   5.   5.]
        [  0.   8.   5. ...,   7.   8.   8.]
        [  0.  10.  10. ...,   8.   8.   8.]]
        [2]   2  [[ 0.  0.  0. ...,  0.  0.  0.]
        [ 0.  2.  2. ...,  6.  5.  5.]
        [ 0.  7.  8. ...,  4.  4.  5.]
        ...,
        [ 0.  4.  1. ...,  4.  5.  6.]
        [ 0.  4.  5. ...,  6.  6.  5.]
        [ 1.  6.  8. ...,  4.  5.  4.]]
        </skip>

    """

    if not isinstance(tags_values_dict, dict):
        raise TypeError("tags_values_dict should be a type of dict, but found type as %" % type(tags_values_dict))

    #Always scala dicom is invoked, as python joins are expensive compared to serailizations.
    def f(scala_dicom):
        scala_dicom.dropRowsByTags(self._tc.jutils.convert.to_scala_map(tags_values_dict))

    self._call_scala(f)