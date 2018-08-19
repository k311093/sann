n = cirno.network {
    cirno.layer { type='input', sx=2, act='tanh' },
    cirno.layer { type='fc', out=20, act='tanh', drop=0.5 },
    cirno.layer { type='fc', out=20, act='tanh', drop=0.5 },
    cirno.layer { type='fc', out=20, act='tanh', drop=0.5 },
    cirno.layer { type='regression'},

    cirno.trainer { method='adagrad', lr=0.01, batch=1000,
        l1_decay = 0, l2_decay = 0, momentum=0.8 }
}

loss = 9999999
epsilon = 0.025
epoch = 1000

while loss / (epoch * 4) > epsilon do
    loss = 0.0

    for i=1,epoch do
        loss = loss + n:train({0, 0}, {0})
        loss = loss + n:train({0, 1}, {1})
        loss = loss + n:train({1, 0}, {1})
        loss = loss + n:train({1, 1}, {0})
    end

    print(loss)
end

print("0 0 = " .. n:predict({0, 0})[1])
print("0 1 = " .. n:predict({0, 1})[1])
print("1 0 = " .. n:predict({1, 0})[1])
print("1 1 = " .. n:predict({1, 1})[1])
