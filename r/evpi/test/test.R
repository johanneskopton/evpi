library(evpi)

x = data.matrix(read.csv("../test_data/x.csv", row.names=1))
y = data.matrix(read.csv("../test_data/y.csv", row.names=1))

multi_evppi = evpi::multi_evppi(x, y)
target_multi_evppi = c(7.1, 2.3, 9.9)
isTRUE(all.equal(multi_evppi, target_multi_evppi, tolerance=0.5))


binary_multi_evppi = evpi::binary_multi_evppi(x, y[,1])
target_binary_multi_evppi = 5.9
isTRUE(all.equal(binary_multi_evppi[1], target_binary_multi_evppi, tolerance=0.5))