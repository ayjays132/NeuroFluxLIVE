from analysis.prompt_optimizer import PromptOptimizer
from unittest.mock import patch


def test_generate_variations():
    with patch('analysis.prompt_optimizer.pipeline') as pipe_mock:
        pipe_mock.return_value = (
            lambda prompt, num_return_sequences, max_new_tokens, do_sample=True, temperature=1.0: [
            {"generated_text": prompt + " variant 1"},
            {"generated_text": prompt + " variant 2"},
            ]
        )
        opt = PromptOptimizer('distilgpt2')
        variations = opt.generate_variations('Test prompt', n_variations=2)
    assert len(variations) == 2
    assert variations[0].startswith('Test prompt')


def test_optimize_prompt():
    opt = PromptOptimizer('distilgpt2')
    with patch.object(opt, 'generate_variations', return_value=['a', 'b']), \
         patch.object(opt, 'score_prompt', side_effect=[2.0, 1.0]):
        best = opt.optimize_prompt('base')
    assert best == 'b'
