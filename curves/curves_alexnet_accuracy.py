import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

X=[]

for k in range(1,201):
    X.append(k)
    
Yalexnetvalaccuracyadadelta=[0.33469998836517334, 0.5001999735832214, 0.5288000106811523, 0.6409000158309937, 0.5824000239372253, 0.6905999779701233, 0.7422999739646912, 0.7369999885559082, 0.7759000062942505, 0.7681999802589417, 0.7423999905586243, 0.8130000233650208, 0.7958999872207642, 0.8051000237464905, 0.8255000114440918, 0.8228999972343445, 0.8375999927520752, 0.8248000144958496, 0.8324000239372253, 0.8345999717712402, 0.8374000191688538, 0.8371000289916992, 0.8367000222206116, 0.8465999960899353, 0.8389999866485596, 0.855400025844574, 0.8431000113487244, 0.8456000089645386, 0.843500018119812, 0.8571000099182129, 0.8457000255584717, 0.8454999923706055, 0.8485000133514404, 0.8529000282287598, 0.8414000272750854, 0.843500018119812, 0.8533999919891357, 0.8569999933242798, 0.8422999978065491, 0.8497999906539917, 0.8565000295639038, 0.8549000024795532, 0.8496000170707703, 0.8574000000953674, 0.8592000007629395, 0.8586000204086304, 0.8537999987602234, 0.859000027179718, 0.8590999841690063, 0.8593999743461609, 0.8597999811172485, 0.8648999929428101, 0.8476999998092651, 0.8607000112533569, 0.8604000210762024, 0.8600000143051147, 0.8684999942779541, 0.8668000102043152, 0.8481000065803528, 0.8554999828338623, 0.8579999804496765, 0.8614000082015991, 0.8657000064849854, 0.8680999875068665, 0.8655999898910522, 0.8636999726295471, 0.8701000213623047, 0.8633999824523926, 0.8690000176429749, 0.8587999939918518, 0.8729000091552734, 0.8651999831199646, 0.866599977016449, 0.864799976348877, 0.867900013923645, 0.864799976348877, 0.8669000267982483, 0.862500011920929, 0.8644000291824341, 0.8636999726295471, 0.8689000010490417, 0.864300012588501, 0.8655999898910522, 0.8715999722480774, 0.8740000128746033, 0.8607000112533569, 0.8691999912261963, 0.8673999905586243, 0.858299970626831, 0.8686000108718872, 0.8592000007629395, 0.8673999905586243, 0.8659999966621399, 0.8734999895095825, 0.8467000126838684, 0.8725000023841858, 0.8698999881744385, 0.8661999702453613, 0.8726000189781189, 0.8691999912261963, 0.8644000291824341, 0.8669000267982483, 0.8691999912261963, 0.8715999722480774, 0.8722000122070312, 0.8687000274658203, 0.8726999759674072, 0.8697999715805054, 0.8719000220298767, 0.8700000047683716, 0.8675000071525574, 0.8700000047683716, 0.8655999898910522, 0.8723999857902527, 0.8672000169754028, 0.8704000115394592, 0.8683000206947327, 0.8705999851226807, 0.8707000017166138, 0.8644000291824341, 0.8658000230789185, 0.8603000044822693, 0.8725000023841858, 0.8626999855041504, 0.8738999962806702, 0.8730000257492065, 0.8689000010490417, 0.8671000003814697, 0.8737999796867371, 0.8695999979972839, 0.8679999709129333, 0.8687999844551086, 0.8651000261306763, 0.8745999932289124, 0.8708000183105469, 0.8718000054359436, 0.8705999851226807, 0.8668000102043152, 0.8628000020980835, 0.8743000030517578, 0.8683000206947327, 0.8697999715805054, 0.8644000291824341, 0.8676999807357788, 0.8615000247955322, 0.8822000026702881, 0.8716999888420105, 0.8657000064849854, 0.8709999918937683, 0.8745999932289124, 0.8725000023841858, 0.8682000041007996, 0.873199999332428, 0.8641999959945679, 0.8726000189781189, 0.8695999979972839, 0.8637999892234802, 0.8708000183105469, 0.8695999979972839, 0.873199999332428, 0.8637999892234802, 0.8680999875068665, 0.8751999735832214, 0.8727999925613403, 0.871399998664856, 0.8765000104904175, 0.8727999925613403, 0.8518000245094299, 0.8729000091552734, 0.8669999837875366, 0.8718000054359436, 0.8740000128746033, 0.8644000291824341, 0.8751000165939331, 0.8677999973297119, 0.8669999837875366, 0.8592000007629395, 0.871399998664856, 0.8741000294685364, 0.8651999831199646, 0.8682000041007996, 0.8705999851226807, 0.8748999834060669, 0.8726000189781189, 0.8646000027656555, 0.8716999888420105, 0.8700000047683716, 0.864799976348877, 0.8680999875068665, 0.8708000183105469, 0.8762999773025513, 0.8737999796867371, 0.866599977016449, 0.8690000176429749, 0.8611999750137329, 0.8669000267982483, 0.8741999864578247, 0.8701000213623047, 0.8604000210762024, 0.8677999973297119]

Yalexnetaccuracyadadelta = [0.21142, 0.38894, 0.49432, 0.5644, 0.62604, 0.66874, 0.69994, 0.72516, 0.74742, 0.7646, 0.77836, 0.79076, 0.80504, 0.81316, 0.82538, 0.83262, 0.84134, 0.8491, 0.85634, 0.8597, 0.8673, 0.8727, 0.87786, 0.88354, 0.88678, 0.8888, 0.89294, 0.89724, 0.89888, 0.90346, 0.90552, 0.90872, 0.91174, 0.91258, 0.91796, 0.9185, 0.91978, 0.92572, 0.92454, 0.92548, 0.92768, 0.92972, 0.93012, 0.93038, 0.9332, 0.93456, 0.93666, 0.93778, 0.93796, 0.93926, 0.94018, 0.94098, 0.9437, 0.9427, 0.9456, 0.94742, 0.94604, 0.94948, 0.94726, 0.9508, 0.95034, 0.95264, 0.95238, 0.95224, 0.95248, 0.9525, 0.9527, 0.95424, 0.95428, 0.95628, 0.95668, 0.95556, 0.95696, 0.95784, 0.9575, 0.95798, 0.9597, 0.95932, 0.9583, 0.95914, 0.96056, 0.96082, 0.9605, 0.96304, 0.9608, 0.96224, 0.96158, 0.96312, 0.96394, 0.9635, 0.96326, 0.96542, 0.96522, 0.9651, 0.96444, 0.965, 0.96554, 0.96458, 0.9639, 0.96574, 0.9662, 0.96642, 0.96662, 0.9675, 0.96604, 0.96746, 0.96622, 0.9678, 0.96882, 0.9672, 0.96832, 0.96704, 0.96854, 0.96998, 0.96946, 0.97006, 0.96814, 0.96884, 0.96956, 0.97078, 0.96986, 0.96978, 0.97036, 0.97124, 0.96932, 0.97072, 0.97014, 0.97102, 0.97148, 0.96952, 0.97266, 0.9726, 0.9721, 0.97352, 0.971, 0.97052, 0.97174, 0.97256, 0.972, 0.97248, 0.971, 0.9733, 0.97322, 0.972, 0.97358, 0.97244, 0.97406, 0.97222, 0.9747, 0.97244, 0.97192, 0.97198, 0.9743, 0.9723, 0.97422, 0.9728, 0.973, 0.97418, 0.97346, 0.97372, 0.9721, 0.97564, 0.97516, 0.97506, 0.97586, 0.9732, 0.97474, 0.9748, 0.97464, 0.9741, 0.97464, 0.97548, 0.97516, 0.974, 0.9761, 0.97562, 0.97386, 0.9751, 0.97652, 0.97528, 0.9749, 0.97572, 0.9754, 0.97642, 0.975, 0.97562, 0.97564, 0.97626, 0.977, 0.97578, 0.9756, 0.97566, 0.97724, 0.97466, 0.9755, 0.97492, 0.97608, 0.97712, 0.97576, 0.97836]

Yalexnetvalaccuracyadam = [0.47540000081062317, 0.5597000122070312, 0.630299985408783, 0.6866999864578247, 0.7055000066757202, 0.7106999754905701, 0.7422999739646912, 0.7523000240325928, 0.7695000171661377, 0.7735000252723694, 0.7696999907493591, 0.7681000232696533, 0.777899980545044, 0.8059999942779541, 0.7994999885559082, 0.7770000100135803, 0.8082000017166138, 0.7921000123023987, 0.805899977684021, 0.8043000102043152, 0.8152999877929688, 0.8180999755859375, 0.8069000244140625, 0.8062000274658203, 0.828499972820282, 0.8187999725341797, 0.8080999851226807, 0.8291000127792358, 0.8235999941825867, 0.8105999827384949, 0.8310999870300293, 0.828499972820282, 0.8256999850273132, 0.8274000287055969, 0.8281999826431274, 0.8320000171661377, 0.832099974155426, 0.8356000185012817, 0.8389000296592712, 0.8325999975204468, 0.8352000117301941, 0.847000002861023, 0.8238000273704529, 0.8141000270843506, 0.830299973487854, 0.8409000039100647, 0.8341000080108643, 0.8388000130653381, 0.8327000141143799, 0.8428999781608582, 0.8363999724388123, 0.8400999903678894, 0.8324000239372253, 0.8424999713897705, 0.8378000259399414, 0.8345000147819519, 0.8500999808311462, 0.849399983882904, 0.8349999785423279, 0.8544999957084656, 0.847100019454956, 0.828000009059906, 0.8449000120162964, 0.8529000282287598, 0.8267999887466431, 0.8501999974250793, 0.8472999930381775, 0.8402000069618225, 0.8536999821662903, 0.8425999879837036, 0.8478000164031982, 0.8443999886512756, 0.8547999858856201, 0.8467000126838684, 0.853600025177002, 0.8508999943733215, 0.8537999987602234, 0.8461999893188477, 0.852400004863739, 0.8490999937057495, 0.8557000160217285, 0.8532000184059143, 0.8474000096321106, 0.8483999967575073, 0.8341000080108643, 0.8540999889373779, 0.8521000146865845, 0.857699990272522, 0.8500000238418579, 0.85589998960495, 0.8500000238418579, 0.8536999821662903, 0.8518000245094299, 0.8517000079154968, 0.849399983882904, 0.8550000190734863, 0.853600025177002, 0.847000002861023, 0.8518000245094299, 0.8600999712944031, 0.8460000157356262, 0.8555999994277954, 0.8464000225067139, 0.8565999865531921, 0.8503999710083008, 0.8495000004768372, 0.8555999994277954, 0.8472999930381775, 0.852400004863739, 0.8513000011444092, 0.8562999963760376, 0.8519999980926514, 0.8497999906539917, 0.8517000079154968, 0.8467000126838684, 0.850600004196167, 0.855400025844574, 0.8569999933242798, 0.8621000051498413, 0.8539000153541565, 0.8571000099182129, 0.857699990272522, 0.8481000065803528, 0.8537999987602234, 0.8514999747276306, 0.8478999733924866, 0.8583999872207642, 0.8495000004768372, 0.8587999939918518, 0.8518999814987183, 0.8547999858856201, 0.8600000143051147, 0.8600999712944031, 0.8604000210762024, 0.855400025844574, 0.8597000241279602, 0.858299970626831, 0.8537999987602234, 0.8604999780654907, 0.8586000204086304, 0.8503999710083008, 0.8529999852180481, 0.8600000143051147, 0.8525000214576721, 0.8492000102996826, 0.8452000021934509, 0.8587999939918518, 0.8517000079154968, 0.8518999814987183, 0.8521999716758728, 0.8601999878883362, 0.8465999960899353, 0.8525000214576721, 0.8585000038146973, 0.8536999821662903, 0.8507000207901001, 0.8517000079154968, 0.8547000288963318, 0.8514000177383423, 0.8589000105857849, 0.8666999936103821, 0.8511999845504761, 0.8547999858856201, 0.8479999899864197, 0.8604000210762024, 0.8582000136375427, 0.8665000200271606, 0.8651000261306763, 0.8495000004768372, 0.8644000291824341, 0.86080002784729, 0.8446999788284302, 0.855400025844574, 0.859000027179718, 0.8575000166893005, 0.8619999885559082, 0.8587999939918518, 0.864799976348877, 0.8562999963760376, 0.8546000123023987, 0.857699990272522, 0.8590999841690063, 0.8629000186920166, 0.8583999872207642, 0.8575999736785889, 0.8634999990463257, 0.8633000254631042, 0.8597999811172485, 0.8586000204086304, 0.8496000170707703, 0.8389999866485596, 0.8629999756813049, 0.8644000291824341, 0.8579000234603882, 0.8183000087738037, 0.8504999876022339, 0.8532000184059143, 0.864799976348877, 0.8611000180244446, 0.8510000109672546]

Yalexnetaccuracyadam = [0.35786, 0.52576, 0.5931, 0.64112, 0.67266, 0.69714, 0.72094, 0.73542, 0.74748, 0.76066, 0.76424, 0.77486, 0.78736, 0.78638, 0.79636, 0.80056, 0.808, 0.80882, 0.81334, 0.81896, 0.81984, 0.82336, 0.82792, 0.8332, 0.83498, 0.83766, 0.84038, 0.84124, 0.847, 0.84818, 0.84828, 0.8505, 0.85346, 0.85676, 0.85684, 0.85712, 0.85976, 0.86118, 0.86492, 0.86548, 0.86604, 0.8669, 0.86968, 0.87072, 0.86826, 0.87488, 0.87628, 0.87488, 0.87548, 0.87986, 0.87734, 0.8823, 0.88162, 0.8817, 0.8832, 0.88606, 0.88626, 0.88544, 0.88756, 0.88842, 0.89026, 0.88962, 0.88966, 0.89308, 0.89192, 0.89284, 0.8959, 0.89624, 0.89628, 0.8988, 0.89834, 0.90198, 0.90044, 0.90302, 0.90144, 0.90024, 0.9046, 0.90464, 0.9056, 0.90618, 0.90762, 0.90626, 0.90762, 0.90818, 0.91108, 0.91042, 0.91194, 0.91306, 0.9132, 0.91444, 0.91452, 0.9135, 0.91208, 0.91594, 0.91354, 0.9193, 0.91334, 0.91724, 0.91662, 0.91942, 0.91882, 0.91882, 0.91874, 0.91898, 0.91792, 0.92124, 0.9206, 0.92084, 0.9196, 0.92008, 0.92204, 0.92448, 0.92608, 0.92194, 0.92574, 0.92476, 0.91948, 0.92438, 0.92676, 0.9251, 0.92292, 0.92722, 0.92844, 0.92558, 0.9246, 0.92602, 0.92882, 0.92738, 0.9276, 0.93154, 0.92946, 0.92866, 0.93114, 0.93282, 0.93316, 0.9271, 0.92526, 0.93056, 0.92872, 0.93292, 0.93112, 0.93094, 0.92946, 0.93096, 0.92498, 0.93538, 0.93224, 0.93308, 0.92834, 0.93044, 0.9329, 0.93056, 0.93364, 0.93406, 0.93694, 0.9343, 0.9355, 0.9348, 0.91838, 0.93386, 0.93546, 0.93592, 0.93802, 0.93302, 0.93378, 0.92808, 0.93204, 0.93462, 0.9323, 0.92444, 0.93762, 0.93252, 0.93116, 0.93812, 0.93606, 0.9357, 0.93178, 0.93974, 0.93718, 0.932, 0.93416, 0.92494, 0.93852, 0.93118, 0.93512, 0.9378, 0.9414, 0.93448, 0.9399, 0.93548, 0.93532, 0.93054, 0.90666, 0.93884, 0.93488, 0.9272, 0.93674, 0.9409, 0.92596, 0.94006]

Yalexnetvalaccuracysgdcustom=[0.10559999942779541, 0.3864000141620636, 0.4620000123977661, 0.4909000098705292, 0.5558000206947327, 0.5999000072479248, 0.6281999945640564, 0.6467000246047974, 0.6722999811172485, 0.6740999817848206, 0.6850000023841858, 0.7299000024795532, 0.7168999910354614, 0.7529000043869019, 0.7541999816894531, 0.7731999754905701, 0.7857000231742859, 0.7770000100135803, 0.7814000248908997, 0.7904000282287598, 0.7588000297546387, 0.8027999997138977, 0.7985000014305115, 0.7962999939918518, 0.8075000047683716, 0.8201000094413757, 0.8116999864578247, 0.8145999908447266, 0.8270000219345093, 0.8269000053405762, 0.8296999931335449, 0.8054999709129333, 0.8285999894142151, 0.8288999795913696, 0.8342999815940857, 0.8370000123977661, 0.8327999711036682, 0.8260999917984009, 0.8409000039100647, 0.8363999724388123, 0.8384000062942505, 0.8410000205039978, 0.8422999978065491, 0.8398000001907349, 0.8432999849319458, 0.8402000069618225, 0.8467000126838684, 0.8306000232696533, 0.8497999906539917, 0.84170001745224, 0.8472999930381775, 0.8503999710083008, 0.8521999716758728, 0.8421000242233276, 0.852400004863739, 0.8539000153541565, 0.8503000140190125, 0.8513000011444092, 0.8424999713897705, 0.8614000082015991, 0.8629999756813049, 0.8604999780654907, 0.8550999760627747, 0.8633000254631042, 0.8496999740600586, 0.8615000247955322, 0.8526999950408936, 0.8562999963760376, 0.8550999760627747, 0.8458999991416931, 0.8396000266075134, 0.8600999712944031, 0.8575000166893005, 0.8579000234603882, 0.8565999865531921, 0.8615000247955322, 0.857699990272522, 0.8623999953269958, 0.8646000027656555, 0.8565999865531921, 0.8574000000953674, 0.8657000064849854, 0.8535000085830688, 0.850600004196167, 0.8616999983787537, 0.8633000254631042, 0.8616999983787537, 0.8596000075340271, 0.8583999872207642, 0.8669000267982483, 0.8677999973297119, 0.8622999787330627, 0.8628000020980835, 0.8651999831199646, 0.8626000285148621, 0.8655999898910522, 0.8640999794006348, 0.8687999844551086, 0.8651000261306763, 0.8634999990463257, 0.8701000213623047, 0.8597000241279602, 0.8608999848365784, 0.8650000095367432, 0.8652999997138977, 0.8668000102043152, 0.8586999773979187, 0.8655999898910522, 0.8702999949455261, 0.8687000274658203, 0.864300012588501, 0.8669000267982483, 0.8715999722480774, 0.8646000027656555, 0.8665000200271606, 0.869700014591217, 0.8672000169754028, 0.8695999979972839, 0.8725000023841858, 0.8634999990463257, 0.8722000122070312, 0.8690999746322632, 0.8669000267982483, 0.8697999715805054, 0.8686000108718872, 0.866100013256073, 0.8676999807357788, 0.8686000108718872, 0.8640999794006348, 0.8671000003814697, 0.8693000078201294, 0.8662999868392944, 0.8690000176429749, 0.8676000237464905, 0.8641999959945679, 0.8675000071525574, 0.8664000034332275, 0.8719000220298767, 0.8730999827384949, 0.8708999752998352, 0.8701000213623047, 0.8712000250816345, 0.8690000176429749, 0.8708000183105469, 0.8733000159263611, 0.8626000285148621, 0.8666999936103821, 0.871999979019165, 0.8677999973297119, 0.8763999938964844, 0.8668000102043152, 0.864300012588501, 0.8730000257492065, 0.8705000281333923, 0.8705000281333923, 0.8672999739646912, 0.8718000054359436, 0.8687999844551086, 0.8697999715805054, 0.8708000183105469, 0.8708999752998352, 0.8769000172615051, 0.8680999875068665, 0.8743000030517578, 0.8672000169754028, 0.8747000098228455, 0.8736000061035156, 0.8755999803543091, 0.8763999938964844, 0.8743000030517578, 0.8716999888420105, 0.8680999875068665, 0.8745999932289124, 0.8740000128746033, 0.8701000213623047, 0.8723999857902527, 0.8762000203132629, 0.8755000233650208, 0.8691999912261963, 0.8690999746322632, 0.8745999932289124, 0.8730000257492065, 0.8708999752998352, 0.8755999803543091, 0.8770999908447266, 0.8762999773025513, 0.8769000172615051, 0.8727999925613403, 0.8761000037193298, 0.8770999908447266, 0.8722000122070312, 0.8759999871253967, 0.878000020980835, 0.8719000220298767, 0.8743000030517578, 0.8736000061035156, 0.8748999834060669, 0.8738999962806702, 0.8636000156402588, 0.8726999759674072]

Yalexnetaccuracysgdcustom = [0.13742, 0.26906, 0.38068, 0.44346, 0.4922, 0.52744, 0.56284, 0.5967, 0.62368, 0.64882, 0.6699, 0.68804, 0.7077, 0.72206, 0.73528, 0.74672, 0.75644, 0.76616, 0.77918, 0.78612, 0.79292, 0.80276, 0.80846, 0.8147, 0.82044, 0.82618, 0.83216, 0.83904, 0.84088, 0.84906, 0.85204, 0.85754, 0.8619, 0.8662, 0.87088, 0.87666, 0.87862, 0.8814, 0.88616, 0.88846, 0.89132, 0.8948, 0.8951, 0.89976, 0.90104, 0.90354, 0.90762, 0.90876, 0.91042, 0.91446, 0.91876, 0.9214, 0.921, 0.92082, 0.92412, 0.92772, 0.92886, 0.92974, 0.93018, 0.93238, 0.93392, 0.93598, 0.93836, 0.93856, 0.94086, 0.94144, 0.94412, 0.94174, 0.94606, 0.94636, 0.9445, 0.94926, 0.94758, 0.95004, 0.95344, 0.952, 0.95414, 0.9532, 0.9559, 0.95618, 0.95786, 0.95808, 0.95932, 0.96264, 0.96102, 0.96222, 0.96302, 0.96096, 0.96492, 0.96318, 0.96268, 0.96496, 0.96496, 0.9659, 0.9661, 0.96744, 0.9684, 0.96764, 0.97086, 0.96944, 0.97026, 0.97138, 0.96912, 0.97254, 0.97096, 0.9719, 0.97138, 0.97266, 0.9719, 0.97184, 0.97524, 0.9733, 0.9749, 0.9744, 0.97506, 0.9754, 0.97732, 0.97528, 0.97546, 0.97462, 0.97656, 0.9776, 0.9783, 0.97906, 0.9799, 0.9778, 0.97922, 0.98014, 0.98026, 0.9789, 0.97982, 0.98094, 0.98048, 0.98082, 0.97982, 0.98182, 0.98148, 0.98204, 0.98264, 0.98286, 0.98324, 0.98292, 0.9831, 0.98242, 0.9824, 0.9834, 0.98326, 0.98336, 0.983, 0.9834, 0.98542, 0.98324, 0.98484, 0.98542, 0.98458, 0.98502, 0.98392, 0.9853, 0.986, 0.98572, 0.98548, 0.9874, 0.98614, 0.98506, 0.9859, 0.98606, 0.98626, 0.98606, 0.9869, 0.98732, 0.98642, 0.98748, 0.98582, 0.98752, 0.98816, 0.98818, 0.98776, 0.98828, 0.98726, 0.98704, 0.98894, 0.98906, 0.98888, 0.98744, 0.9882, 0.98822, 0.99028, 0.98788, 0.98998, 0.98862, 0.98752, 0.98912, 0.98976, 0.98956, 0.98938, 0.98918, 0.98818, 0.98912, 0.99022, 0.9901]
plt.figure()

plt.plot(X,Yalexnetaccuracyadadelta)

plt.plot(X,Yalexnetaccuracyadam)

plt.plot(X,Yalexnetaccuracysgdcustom)
plt.title("alexnet")
plt.legend(["acc_adadelta","acc_adam","acc_sgdcustom"])
plt.savefig('accuracy_alexnet.png')
plt.show()