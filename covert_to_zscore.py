import click
import pandas as pd

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
def calculate_z_scores(input_file, output_file):
    """计算输入文件中每个浮点数的z-score，并将结果输出到指定的输出文件中。"""
    # 读取文件
    data = pd.read_csv(input_file, header=None, names=["Value"])
    
    # 计算z-score
    mean = data['Value'].mean()
    std = data['Value'].std()
    data['Z-Score'] = (data['Value'] - mean) / std
    
    # 输出到文件
    data['Z-Score'].to_csv(output_file, index=False, header=False)

    # 打印一个完成消息
    click.echo(f"Z-scores have been calculated and saved to {output_file}")

if __name__ == '__main__':
    calculate_z_scores()
