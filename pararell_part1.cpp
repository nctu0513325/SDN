void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  __pp_vec_float x, result, maxResult;
  __pp_vec_int y, count;
  __pp_mask maskAll, maskIsZero, maskIsNotZero, maskClamp;

  // 設定最大結果值為 9.999999
  maxResult = _pp_vset_float(9.999999f);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // 初始化掩碼
    maskAll = _pp_init_ones();
    maskIsZero = _pp_init_ones(0);

    // 從連續的記憶體地址載入向量值
    _pp_vload_float(x, values + i, maskAll);
    _pp_vload_int(y, exponents + i, maskAll);

    // 設定掩碼根據指數是否為零
    _pp_veq_int(maskIsZero, y, _pp_vset_int(0), maskAll);
    _pp_vset_float(result, 1.0f, maskIsZero);

    // 反轉掩碼以生成 "非零" 掩碼
    maskIsNotZero = _pp_mask_not(maskIsZero);
    _pp_vmove_float(result, x, maskIsNotZero);

    // 計數減一
    _pp_vsub_int(count, y, _pp_vset_int(1), maskIsNotZero);

    // 當 count > 0 時，繼續乘法運算
    while (_pp_cntbits(maskIsNotZero) > 0)
    {
      _pp_vmult_float(result, result, x, maskIsNotZero);
      _pp_vsub_int(count, count, _pp_vset_int(1), maskIsNotZero);
      _pp_vgt_int(maskIsNotZero, count, _pp_vset_int(0), maskAll);
    }

    // 將結果與最大值進行比較並進行夾取
    _pp_vgt_float(maskClamp, result, maxResult, maskAll);
    _pp_vmove_float(result, maxResult, maskClamp);

    // 將結果寫回記憶體
    _pp_vstore_float(output + i, result, maskAll);
  }
}
float arraySumVector(float *values, int N)
{
  __pp_vec_float x, sum;
  __pp_mask maskAll = _pp_init_ones();

  // 初始化總和向量為 0
  sum = _pp_vset_float(0.0f);

  // 遍歷數組，以 VECTOR_WIDTH 為步長
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // 從 values 數組中加載當前批次的浮點數到向量 x 中
    _pp_vload_float(x, values + i, maskAll);
    // 將 x 向量中的元素累加到 sum 向量中
    _pp_vadd_float(sum, sum, x, maskAll);
  }

  // 向量內部求和，將 sum 向量中的所有元素相加
  for (int width = VECTOR_WIDTH; width > 1; width /= 2)
  {
    __pp_vec_float temp;
    // 水平加法，將相鄰元素相加
    _pp_hadd_float(sum, sum);
    // 交錯存儲，將加法結果存儲到 temp 向量中
    _pp_interleave_float(temp, sum);
    // 將 temp 向量中的值移動到 sum 向量中
    _pp_vmove_float(sum, temp, maskAll);
  }

  // 將最終的總和存儲到 result 中
  float result;
  _pp_vstore_float(&result, sum, maskAll);

  return result;
}
