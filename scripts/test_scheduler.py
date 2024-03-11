from thinc.api import compounding, warmup_linear  

max = 8
min = 2
rate = 1.0005

batch_sizes = compounding(min, max, rate)

for n, batch_sizes in enumerate(batch_sizes):
    print(batch_sizes)
    if batch_sizes >= max:
        print(n)
        break


initial_rate = 0.00005
warmup_steps = 2000
total_steps = 20000
learn_rates = warmup_linear(initial_rate, warmup_steps, total_steps)

for n, lr in enumerate(learn_rates):
    if n % 100 == 0:
        print(f"Learning rate at {n}: {lr}")
    elif n > 2000:
        break