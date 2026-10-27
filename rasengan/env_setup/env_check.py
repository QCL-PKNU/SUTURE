from rasengan.model import LinearConstrainedBinaryOptimization as LcboModel
from rasengan.solvers.optimizers import CobylaOptimizer, AdamOptimizer
from rasengan.solvers.qiskit import (
    HeaSolver, PenaltySolver, CyclicSolver, ChocoSolver, RasenganSolver,
    AerGpuProvider, AerProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, DdsimProvider,
)

# model ----------------------------------------------
m = LcboModel()
x = m.addVars(3, name="x")
m.setObjective(x[0] + x[1] - x[2], "max")
m.addConstr(x[0] + x[1] + x[2] == 2)
optimize = m.optimize()
# sovler ----------------------------------------------
opt = CobylaOptimizer(max_iter=200)
gpu = AerGpuProvider()
aer = DdsimProvider()

solver_cpu = RasenganSolver(
    prb_model=m,
    optimizer=opt,
    provider=aer,
    num_layers=1,
)
print("="*50)
try:
    result = solver_cpu.solve()
    eval = solver_cpu.evaluation()
    # print(eval)
    print("✅  CPU environment configured successfully!")
except Exception as e:
    print("❌  CPU environment configuration failed.")
    print(e)

solver_gpu = RasenganSolver(
    prb_model=m,
    optimizer=opt,
    provider=gpu,
    num_layers=1,
)

try:
    result = solver_gpu.solve()
    eval = solver_gpu.evaluation()
    # print(eval)
    print("✅  GPU environment configured successfully!")
except Exception as e:
    print("❌  GPU environment configuration failed.")
    print(e)
print("="*50)

