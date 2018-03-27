import tensorflow as tf
import numpy as np


s2pi = 2.50662827463100050242E0

P0 = [
      -5.99633501014107895267E1,
      9.80010754185999661536E1,
      -5.66762857469070293439E1,
      1.39312609387279679503E1,
      -1.23916583867381258016E0,
      ]

Q0 = [
      1,
      1.95448858338141759834E0,
      4.67627912898881538453E0,
      8.63602421390890590575E1,
      -2.25462687854119370527E2,
      2.00260212380060660359E2,
      -8.20372256168333339912E1,
      1.59056225126211695515E1,
      -1.18331621121330003142E0,
      ]

P1 = [
      4.05544892305962419923E0,
      3.15251094599893866154E1,
      5.71628192246421288162E1,
      4.40805073893200834700E1,
      1.46849561928858024014E1,
      2.18663306850790267539E0,
      -1.40256079171354495875E-1,
      -3.50424626827848203418E-2,
      -8.57456785154685413611E-4,
      ]

Q1 = [
      1,
      1.57799883256466749731E1,
      4.53907635128879210584E1,
      4.13172038254672030440E1,
      1.50425385692907503408E1,
      2.50464946208309415979E0,
      -1.42182922854787788574E-1,
      -3.80806407691578277194E-2,
      -9.33259480895457427372E-4,
      ]

P2 = [
      3.23774891776946035970E0,
      6.91522889068984211695E0,
      3.93881025292474443415E0,
      1.33303460815807542389E0,
      2.01485389549179081538E-1,
      1.23716634817820021358E-2,
      3.01581553508235416007E-4,
      2.65806974686737550832E-6,
      6.23974539184983293730E-9,
      ]

Q2 = [
      1,
      6.02427039364742014255E0,
      3.67983563856160859403E0,
      1.37702099489081330271E0,
      2.16236993594496635890E-1,
      1.34204006088543189037E-2,
      3.28014464682127739104E-4,
      2.89247864745380683936E-6,
      6.79019408009981274425E-9,
      ]


def tfppf(y0):
    dtype = y0.dtype
    tfy0 = y0
    if len(tfy0.get_shape().as_list()) == 0:
        tfy0 = tf.expand_dims(tfy0, -1)

    tfy = tfy0
    c = tf.constant(0.13533528323661269189, dtype=dtype)
    one = tf.constant(1.0, dtype=dtype)
    ones = tf.ones(tf.shape(tfy0), dtype=dtype)
    tfy = tf.where(tf.greater(tfy, one - c), one - tfy, tfy)

    tfnegate = tf.where(
        tf.greater(tfy0, (one - c)*ones),
        tf.zeros(tf.shape(tfy0), dtype=dtype),
        tf.ones(tf.shape(tfy0), dtype=dtype))

    tfy1 = tfy - 0.5
    tfy2 = tfy1**2
    tfx = tfy1 + tfy1 * (tfy2 * polevl(tfy2, P0) / polevl(tfy2, Q0))
    tfx = tfx * s2pi

    tfx = tf.where(tf.less_equal(tfy, c),
                   tf.sqrt(-2.0 * tf.log(tfy)),
                   tfx)
    tfx0 = tfx - tf.log(tfx) / tfx

    tfz = 1./tfx

    tfx1 = tfz * polevl(tfz, P1) / polevl(tfz, Q1)
    tfx1 = tf.where(tf.greater_equal(tfx, 8.0),
                    tfz * polevl(tfz, P2) / polevl(tfz, Q2),
                    tfx1)

    tfx = tf.where(tf.less_equal(tfy, c),
                   tfx0 - tfx1,
                   tfx)
    tfx = tf.where(
        tf.logical_and(tf.less_equal(tfy, c), tf.equal(tfnegate, one)),
        -tfx,
        tfx)
    tfx = tf.where(tf.equal(tfy0, 0.), -np.inf*ones, tfx)
    tfx = tf.where(tf.equal(tfy0, 1.), np.inf*ones, tfx)

    return(tfx)


def polevl(x, coef):
    accum = 0
    for c in coef:
        accum = x * accum + c
    return(accum)
