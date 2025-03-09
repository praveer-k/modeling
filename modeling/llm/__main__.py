import mlx.core as mx
from click import group
from modeling.llm.mlx__model import Llama

@group("llm")
def cli():
    """Large Language Model CLI Example"""
    pass

@cli.command()
def generate():
    """generate example"""
    model = Llama(num_layers=12, vocab_size=8192, dims=512, mlp_dims=1024, num_heads=8)
    # Since MLX is lazily evaluated nothing has actually been materialized yet.
    # We could have set the `dims` to 20_000 on a machine with 8GB of RAM and the
    # code above would still run. Let's actually materialize the model.
    mx.eval(model.parameters())
    prompt = mx.array([[1, 10, 8, 32, 44, 7]])  # <-- Note the double brackets because we
                                                #     have a batch dimension even
                                                #     though it is 1 in this case
    generated = [t for i, t in zip(range(10), model.generate(prompt, 0.8))]
    # Since we haven't evaluated anything, nothing is computed yet. The list
    # `generated` contains the arrays that hold the computation graph for the
    # full processing of the prompt and the generation of 10 tokens.
    #
    # We can evaluate them one at a time, or all together. Concatenate them or
    # print them. They would all result in very similar runtimes and give exactly
    # the same results.
    mx.eval(generated)
    print(generated)