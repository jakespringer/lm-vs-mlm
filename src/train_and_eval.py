import torch
import algorithms
import json
import logging
import argparse
import numpy as np
import itertools

from model import GPT, GPTConfig
from star_graph import StarGraph, StarGraphTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train_and_eval.py')

def main(args):
    # Load the data and tokenizer
    logger.info("Loading the data")
    graphs = StarGraph.random_graphs(
        args.num_branches, 
        args.branch_length, 
        args.num_train, 
        args.num_vertices, 
        dist='constant')
    tokenizer = StarGraphTokenizer(args.num_vertices)
    tokens = tokenizer.tokenize(graphs, with_solution=True, padding=True, return_tensors='pt')
    train_dataset = algorithms.DictDataset(tokens)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True)
    
    logger.info('Loading the model and trainer')
    causal_objectives = ['lm']
    config = GPTConfig(
        block_size=args.block_size, 
        vocab_size=args.num_vertices + tokenizer.get_special_tokens_count(), 
        n_layer=args.num_layer, 
        n_head=args.num_head, 
        n_embd=args.num_embd,
        dropout=args.dropout,
        bias=args.bias,
        is_causal=args.objective in causal_objectives)
    model = GPT(config)
    model = model.to(args.device)
    if args.objective == 'lm':
        trainer = algorithms.LMTrainer(
            model, 
            train_loader, 
            args.lr, 
            args.wd, 
            pad_token=tokenizer.pad_token_id,
            device=args.device)
    elif args.objective == 'mlm':
        trainer = algorithms.MLMTrainer(
            model, 
            train_loader, 
            args.lr, 
            args.wd, 
            mask_ratio=args.mask_ratio, 
            mask_token=tokenizer.mask_token_id,
            device=args.device)
    else:
        raise ValueError("objective must be one of ['lm', 'mlm']")

    # Train the model
    logger.info("Training the model")
    losses = trainer.train(args.num_epochs)

    # Save the model
    if args.model_path:
        logger.info("Saving the model")
        torch.save({
            'state_dict': model.state_dict(), 
            'config': config.__dict__,
            'args': args,
            'losses': losses,
        }, args.model_path)

    model.eval()

    # Evaluate the model
    logger.info("Evaluating the model")
    graphs = StarGraph.random_graphs(
        args.num_branches, 
        args.branch_length, 
        args.num_test, 
        args.num_vertices, 
        dist='constant')
    eval_tokens_with_solution = tokenizer.tokenize(graphs, with_solution=True, padding=True, return_tensors='pt')
    eval_tokens_without_solution = tokenizer.tokenize(graphs, with_solution='padded', padding=True, return_tensors='pt')
    eval_dataset_with_solution = algorithms.DictDataset(eval_tokens_with_solution)
    eval_dataset_without_solution = algorithms.DictDataset(eval_tokens_without_solution)

    if args.objective == 'lm':
        sampler = algorithms.LMNextTokenArgmaxSampler(model, tokenizer, device=args.device)
        evaluator = algorithms.LMEvaluator(
            model,
            sampler, 
            tokenizer)
        accuracy = evaluator.evaluate(eval_dataset_without_solution, eval_dataset_with_solution)
    elif args.objective == 'mlm':
        sampler = algorithms.MLMLocationAndTokenArgmaxSampler(model, tokenizer, device=args.device)
        evaluator = algorithms.MLMEvaluator(
            model,
            sampler, 
            tokenizer)
        accuracy = evaluator.evaluate(eval_dataset_with_solution, eval_dataset_with_solution)
    else:
        raise ValueError("objective must be one of ['lm', 'mlm']")
    print(f"Accuracy: {accuracy*100:.2f}")
    logger.info("Done")
    # print(losses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_branches', type=int)
    parser.add_argument('--branch_length', type=int)
    parser.add_argument('--num_train', type=int)
    parser.add_argument('--num_test', type=int)
    parser.add_argument('--num_vertices', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_layer', type=int)
    parser.add_argument('--num_head', type=int)
    parser.add_argument('--num_embd', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--bias', type=bool)
    parser.add_argument('--block_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--mask_ratio', type=float)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--objective', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()
    main(args)