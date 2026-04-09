import subjective_logic as sl

test_a = sl.Opinion2d(0.2,0.5)
test_b = sl.Opinion2d(0.8,0.1)
test_c = sl.Opinion2d(0.3,0.3)

test_cum = test_a.cum_fuse(test_b)
print('test_cum is ', test_cum)

multi_cum = sl.Fusion.fuse_opinions(sl.FusionType.CUMULATIVE, [test_a, test_b])
print('multi_cum is ', multi_cum)
print()

test_cum.cum_fuse_(test_c)
print('test_cum2 is ', test_cum)
multi_cum = sl.Fusion.fuse_opinions(sl.FusionType.CUMULATIVE, [test_a, test_b, test_c])
print('multi_cum2 is ', multi_cum)



obs_list = [
    sl.Opinion(0.3, 0.4),
    sl.Opinion(0.7, 0.1),
    sl.Opinion(0.4, 0.1),
    sl.Opinion(0.3, 0.3),
    sl.Opinion(0.4, 0.1),
    sl.Opinion(0.1, 0.1),
    sl.Opinion(0.7, 0.2),
    sl.Opinion(0.4, 0.2),
]
cbf_seq = obs_list[0]
for i in range(1, len(obs_list)):
    cbf_seq.cum_fuse_(obs_list[i])
msf_cbf = sl.Fusion.fuse_opinions(sl.FusionType.CUMULATIVE, obs_list)

print('seq:', cbf_seq)
print('multi:', cbf_seq)
