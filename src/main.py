import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs,MACCSkeys
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import pickle

class AdvancedMolecularPropertyClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        self.y_train = None

    def smiles_to_fingerprint(self, smiles, radius=2, n_bits=2048):
        """
        将SMILES字符串转换为Morgan指纹
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            # fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except:
            return None
    
    def load_and_preprocess_data(self, file_path, smiles_col='SMILES', class_col='class'):
        """
        加载CSV数据并预处理
        """
        print("正在加载数据...")
        # df = pd.read_csv(file_path)
        df = pd.read_excel(file_path)
        
        print(f"原始数据形状: {df.shape}")
        print(f"数据列: {df.columns.tolist()}")
        
        # 检查必要的列是否存在
        if smiles_col not in df.columns:
            raise ValueError(f"列 '{smiles_col}' 不存在于数据中")
        if class_col not in df.columns:
            raise ValueError(f"列 '{class_col}' 不存在于数据中")
        
        # 删除缺失值
        df = df.dropna(subset=[smiles_col, class_col])
        print(f"删除缺失值后数据形状: {df.shape}")
        
        # 编码类别标签
        print("正在编码类别标签...")
        desired_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

        # 使用fit时指定类别
        self.label_encoder.fit(desired_order)
        y_encoded = self.label_encoder.transform(df[class_col])
        class_names = self.label_encoder.classes_

        
        print(f"类别数量: {len(class_names)}")
        print(f"类别名称: {class_names}")
        
        # 转换为分子指纹
        print("正在转换SMILES为分子指纹...")
        fingerprints = []
        valid_indices = []
        
        for idx, smiles in enumerate(df[smiles_col]):
            fp = self.smiles_to_fingerprint(smiles)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)
            else:
                raise('无效的SMILES字符串，索引:', idx, 'SMILES:', smiles)
        fingerprints = np.array(fingerprints)
        classes = y_encoded[valid_indices]
        
        print(f"有效分子数量: {len(fingerprints)}")
        print(f"指纹特征维度: {fingerprints.shape}")
        print(f"类别分布: {pd.Series(classes).value_counts().to_dict()}")
        
        return fingerprints, classes, df.iloc[valid_indices], class_names
    
    def initialize_models(self):
        """
        初始化所有分类模型
        """
        self.models = {
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        }
    
    def hyperparameter_tuning(self, X, y, model_name, param_grid, cv=5):
        """
        超参数调优
        """
        print(f"\n正在对 {model_name} 进行超参数调优...")
        
        if model_name == 'Decision Tree':
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == 'Random Forest':
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_name == 'Gradient Boosting':
            model = GradientBoostingClassifier(random_state=42)
        elif model_name == 'MLP':
            model = MLPClassifier(random_state=42, early_stopping=True)
        else:
            return self.models[model_name]
        
        # 使用网格搜索
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='f1_macro', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X, y)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证 F1-macro: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_all_models(self, X, y, test_size=0.2, random_state=42, tune_hyperparams=False):
        """
        训练所有模型并进行比较
        """
        print("\n正在准备训练数据...")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.y_train = y_train
        
        # print(X_test.shape)
        
        # 标准化特征（对MLP特别重要）
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        print(f"训练集类别分布: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"测试集类别分布: {pd.Series(y_test).value_counts().to_dict()}")
        
        # 初始化模型
        self.initialize_models()
        
        # 超参数网格
        param_grids = {
            'Decision Tree': {
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20],
                'criterion': ['gini', 'entropy']
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 6, 9]
            },
            'MLP': {
                'hidden_layer_sizes': [(50,), (100, 50), (100, 100)],
                'alpha': [0.001, 0.0001],
                'learning_rate_init': [0.001, 0.01]
            }
        }
        
        results = {}
        best_f1 = -np.inf
        
        for model_name, model in self.models.items():
            print(f"\n=== 训练 {model_name} ===")
            # if model_name == "MLP":
            #     continue
            # 超参数调优
            if tune_hyperparams:
                model = self.hyperparameter_tuning(
                    X_train_scaled, y_train, model_name, param_grids[model_name]
                )
                self.models[model_name] = model
            
            # 训练模型
            model.fit(X_train_scaled, y_train)
            
            # 预测
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            #单独画图
            # self._plot_model_comparison()
            # '''
            # results[model_name] = {
            #     'model': model,
            #     'y_test_pred': y_test_pred,
            #     'y_train_pred': y_train_pred,
            #     'y_test_proba': y_test_proba,
            #     'metrics': {
            #         'train_accuracy': train_accuracy,
            #         'test_accuracy': test_accuracy,
            #         'train_precision': train_precision,
            #         'test_precision': test_precision,
            #         'train_recall': train_recall,
            #         'test_recall': test_recall,
            #         'train_f1': train_f1,
            #         'test_f1': test_f1,
            #         'test_auc': auc_score,
            #         'cv_mean': cv_scores.mean(),
            #         'cv_std': cv_scores.std()
            #     }
            # '''


            # 计算概率（用于AUC计算）
            if hasattr(model, "predict_proba"):
                y_test_proba = model.predict_proba(X_test_scaled)
            else:
                y_test_proba = None
            
            # 计算评估指标
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            train_precision = precision_score(y_train, y_train_pred, average='macro', zero_division=0)
            test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
            train_recall = recall_score(y_train, y_train_pred, average='macro', zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
            train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
            
            # 计算AUC（多分类）
            if y_test_proba is not None and len(np.unique(y_test)) > 1:
                if len(np.unique(y_test)) == 2:  # 二分类
                    auc_score = roc_auc_score(y_test, y_test_proba[:, 1])
                else:  # 多分类
                    auc_score = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro')
            else:
                auc_score = np.nan
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                      cv=5, scoring='f1_macro')
            
            results[model_name] = {
                'model': model,
                'y_test_pred': y_test_pred,
                'y_test':y_test,
                'y_train':y_train,
                'y_train_pred': y_train_pred,
                'y_test_proba': y_test_proba,
                'metrics': {
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_precision': train_precision,
                    'test_precision': test_precision,
                    'train_recall': train_recall,
                    'test_recall': test_recall,
                    'train_f1': train_f1,
                    'test_f1': test_f1,
                    'test_auc': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
            }
            
            print(f"测试集准确率: {test_accuracy:.4f}")
            print(f"测试集 F1-macro: {test_f1:.4f}")
            print(f"测试集 AUC: {auc_score:.4f}")
            print(f"交叉验证 F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # 更新最佳模型
            if test_f1 > best_f1:
                best_f1 = test_f1
                self.best_model = model
                self.best_model_name = model_name
        
        # 添加数据集信息到结果中
        results['data'] = {
            'X_train': X_train, 'X_test': X_test,
            'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
            'y_train': y_train, 'y_test': y_test
        }
        
        return results
    
    def compare_models(self, results):
        """
        比较所有模型的性能
        """
        print("\n" + "="*50)
        print("模型性能比较")
        print("="*50)
        
        comparison_data = []
        for model_name, result in results.items():
            if model_name != 'data':
                metrics = result['metrics']
                comparison_data.append({
                    'Model': model_name,
                    'Test Accuracy': f"{metrics['test_accuracy']:.4f}",
                    'Test F1-macro': f"{metrics['test_f1']:.4f}",
                    'Test Precision': f"{metrics['test_precision']:.4f}",
                    'Test Recall': f"{metrics['test_recall']:.4f}",
                    'Test AUC': f"{metrics['test_auc']:.4f}",
                    'Train Accuracy': f"{metrics['train_accuracy']:.4f}",
                    'CV F1-macro': f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}"
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Test F1-macro', ascending=False)
        
        print(df_comparison.to_string(index=False))
        print(f"\n最佳模型: {self.best_model_name}")
        
        return df_comparison
    
    def plot_model_comparison(self, results, class_names):
        """
        绘制模型比较图
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 模型F1分数比较
        models = []
        test_f1_scores = []
        train_f1_scores = []
        
        for model_name, result in results.items():
            if model_name != 'data':
                models.append(model_name)
                test_f1_scores.append(result['metrics']['test_f1'])
                train_f1_scores.append(result['metrics']['train_f1'])
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, train_f1_scores, width, label='training set', alpha=0.7)
        axes[0, 0].bar(x + width/2, test_f1_scores, width, label='test set', alpha=0.7)
        axes[0, 0].set_xlabel('model')
        axes[0, 0].set_ylabel('F1 Score (macro)')
        axes[0, 0].set_title('Model performance comparison (F1 Score)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, )#rotation=45
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 混淆矩阵 (最佳模型)
        best_model_name = self.best_model_name
        y_test_pred = results[best_model_name]['y_test_pred']
        y_test = results['data']['y_test']
        
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
        axes[0, 1].set_xlabel('Predicted label')
        axes[0, 1].set_ylabel('True label')
        axes[0, 1].set_title(f'optimization model ({best_model_name}) confusion matrix\n'
                           f'accuracy rate = {results[best_model_name]["metrics"]["test_accuracy"]:.4f}')
        
        # 3. 多指标比较雷达图
        metrics_to_plot = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        #metric_names = ['precision', 'precision ratio', 'recall rate', 'F1-score model']
        
        # 准备雷达图数据
        angles = np.linspace(0, 2*np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        fig.delaxes(axes[1,0])
        # 在相同位置创建极坐标子图
        ax_radar = fig.add_subplot(2, 2, 3, projection='polar')

        
        for model_name, result in results.items():
            if model_name != 'data':
                values = [result['metrics'][metric] for metric in metrics_to_plot]
                values += values[:1]  # 闭合雷达图
                ax_radar.plot(angles, values, 'o-', linewidth=2, label=model_name)
                ax_radar.fill(angles, values, alpha=0.1)
        
        # ax_radar.set_thetagrids(np.degrees(angles[:-1]), metric_names)
        ax_radar.set_title('Multi-index comparison of the model')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax_radar.grid(True)
        ax_radar.set_yticklabels([]) 

        # 4. 特征重要性（如果可用）
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[-20:]
            axes[1, 1].barh(range(20), importances[indices])
            axes[1, 1].set_yticks(range(20))
            axes[1, 1].set_yticklabels([f'feature {i}' for i in indices])
            axes[1, 1].set_xlabel('feature importance')
            axes[1, 1].set_title(f'optimization model ({best_model_name}) feature importance')
        else:
            # 对于MLP等没有feature_importance的模型，显示准确率比较
            models = []
            accuracy_scores = []
            for model_name, result in results.items():
                if model_name != 'data':
                    models.append(model_name)
                    accuracy_scores.append(result['metrics']['test_accuracy'])
            
            axes[1, 1].bar(models, accuracy_scores, alpha=0.7)
            axes[1, 1].set_xlabel('模型')
            axes[1, 1].set_ylabel('测试集准确率')
            axes[1, 1].set_title('模型准确率比较')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def predict_with_best_model(self, smiles_list):
        """
        使用最佳模型预测新分子
        """
        if self.best_model is None:
            raise ValueError("没有可用的训练模型")
        
        fingerprints = []
        valid_smiles = []
        
        for smiles in smiles_list:
            fp = self.smiles_to_fingerprint(smiles)
            if fp is not None:
                fingerprints.append(fp)
                valid_smiles.append(smiles)
        
        if not fingerprints:
            raise ValueError("没有有效的SMILES字符串")
        
        fingerprints = np.array(fingerprints)
        fingerprints_scaled = self.scaler.transform(fingerprints)
        
        # 预测类别和概率
        predictions = self.best_model.predict(fingerprints_scaled)
        predicted_classes = self.label_encoder.inverse_transform(predictions)
        
        result_df = pd.DataFrame({
            'smiles': valid_smiles,
            'predicted_class': predicted_classes,
            'class_code': predictions,
            'model_used': self.best_model_name
        })
        
        # 如果模型支持概率预测，添加概率
        if hasattr(self.best_model, "predict_proba"):
            probabilities = self.best_model.predict_proba(fingerprints_scaled)
            for i, class_name in enumerate(self.label_encoder.classes_):
                result_df[f'prob_{class_name}'] = probabilities[:, i]
        
        return result_df

    def _plot_model_comparison(self, results, class_names, y_train_pred=False):
        for model_name in self.models:
        # 为每个模型创建一个单独的图
            plt.figure(figsize=(10, 8))
            
            y_test_pred = results[model_name]['y_test_pred']
            y_test = results['data']['y_test']
            mode_ = 'test'

            if y_train_pred:
                y_test_pred = results[model_name]['y_train_pred']
                y_test = results['data']['y_train']
                mode_ = 'training'

            # 计算混淆矩阵
            cm = confusion_matrix(y_test, y_test_pred)
            
            # 绘制热图
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.title(f'Model ({model_name}) Confusion Matrix\n'
                    f'Accuracy Rate = {results[model_name]["metrics"]["test_accuracy"]:.4f}')
            
            plt.tight_layout()
            # plt.show()  # 显示当前图形
            plt.savefig(f'{mode_}_{model_name}.png')
            plt.close()  # 关闭当前 figure, 避免图像累积占用内存
            
        # fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        
        
        # # 2. 混淆矩阵 (最佳模型)
        # for best_model_name in self.models:
        # #best_model_name = self.best_model_name
        #     y_test_pred = results[best_model_name]['y_test_pred']
        #     y_test = results['data']['y_test']
            
        #     cm = confusion_matrix(y_test, y_test_pred)
        #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
        #             xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
        #     axes[0, 1].set_xlabel('Predicted label')
        #     axes[0, 1].set_ylabel('True label')
        #     axes[0, 1].set_title(f'optimization model ({best_model_name}) confusion matrix\n'
        #                     f'accuracy rate = {results[best_model_name]["metrics"]["test_accuracy"]:.4f}')
            
            

        
        
        

def load_model_bundle(path):
    """
    加载 save_bundle 保存的模型包, 返回 (model, scaler, label_encoder, model_name)
    """
    with open(path, 'rb') as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict):
        return (bundle['model'], bundle['scaler'],
                bundle['label_encoder'], bundle['model_name'])

    # 兼容旧格式: 裸模型对象。缺少 scaler, 只能喂已标准化的特征, 否则预测结果错误。
    print(f"警告: {path} 是旧格式(裸模型, 无 scaler), 预测结果可能不正确")
    return bundle, None, None, type(bundle).__name__


def predict_smiles_by_model(path, smiles_list):
    """
    加载模型包并对新分子预测(自动标准化), 供 predict.ipynb 等外部脚本复用
    """
    model, scaler, label_encoder, name = load_model_bundle(path)
    if scaler is None:
        raise ValueError(
            f"{path} 中没有 scaler, 无法对原始指纹预测。请用修复后的 main.py 重新训练并保存。")

    helper = AdvancedMolecularPropertyClassifier()
    fingerprints = []
    valid_smiles = []
    for smiles in smiles_list:
        fp = helper.smiles_to_fingerprint(smiles)
        if fp is None:
            print(f"无效 SMILES, 已跳过: {smiles}")
            continue
        fingerprints.append(fp)
        valid_smiles.append(smiles)

    X = scaler.transform(np.array(fingerprints))
    predictions = model.predict(X)

    result_df = pd.DataFrame({
        'smiles': valid_smiles,
        'predicted_class': label_encoder.inverse_transform(predictions),
        'class_code': predictions,
        'model_used': name,
    })

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        for i, class_name in enumerate(label_encoder.classes_):
            result_df[f'prob_{class_name}'] = probabilities[:, i]

    return result_df


def main():
    """
    主函数 - 分类版本
    """
    print("=== 多模型分子分类预测 Demo ===\n")
    
    # 初始化分类器
    classifier = AdvancedMolecularPropertyClassifier()
    
    file_path = 'data.xlsx'
    
    try:
        # 加载和预处理数据
        X, y, df, class_names = classifier.load_and_preprocess_data(
            file_path, 
            smiles_col='SMILES',
            class_col='class'  # 假设类别列名为'class'
        )
        
        # 显示数据统计
        print(f"\n类别统计:")
        print(f"总样本数量: {len(y)}")
        print(f"类别数量: {len(class_names)}")
        print(f"类别分布: {pd.Series(y).value_counts().to_dict()}")
        
        # 询问是否进行超参数调优
        tune_params = input("\n是否进行超参数调优? (y/n, 默认n): ").strip().lower() == 'y'
        
        # 训练所有模型
        results = classifier.train_all_models(X, y, tune_hyperparams=tune_params)
        
        # 比较模型性能
        comparison_df = classifier.compare_models(results)
        
        # 绘制结果
        classifier.plot_model_comparison(results, class_names)
        
        #绘制各个模型的matrix
        classifier._plot_model_comparison(results, class_names)
        classifier._plot_model_comparison(results, class_names,y_train_pred=True)
        #保存模型
        #注意: 模型是在标准化后的特征上训练的, 树的阈值位于标准化坐标系,
        #      单独保存模型会导致加载后用原始指纹预测时结果完全错误,
        #      因此必须把 scaler / label_encoder 一起打包保存。
        def save_bundle(path, name, model):
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': classifier.scaler,
                    'label_encoder': classifier.label_encoder,
                    'model_name': name,
                    'class_names': list(class_names),
                    'n_features': 167,          # MACCS keys
                }, f)
            print(f"已保存: {path}  ({name})")

        #保存性能最好的模型
        save_bundle('best_model.pkl', classifier.best_model_name, classifier.best_model)

        #保存所有的模型
        #注意: 必须用 .items() 取到模型对象, 直接用 for model in dict 拿到的是 key(字符串)
        for name, model in classifier.models.items():
            save_bundle(f'{name}.pkl', name, model)

        # 演示预测新分子
        print("\n=== 新分子预测演示 ===")
        # test_df = pd.read_excel('yanzhengdata.xlsx')
        test_smiles = ['CC1=CC[C@@H](OC1=O)[C@@H](C)[C@H]2CC[C@@]3([C@@]2(CCC4=C3CC[C@@H]5C(=C4)C=CC(=O)OC5(C)C)C)C', 
                       'O[C@]12[C@]3([H])[C@@](C)([C@@]3([C@H](C)[C@H](OC(C)=O)[C@]4([H])C=C(C)C(O4)=O)[H])CC[C@@]([C@H](O)[C@](OC5=O)([C@@]([H])(C5)OC6(C)C)[C@@]6([H])CC7)(O2)[C@]7(O)C1=O',
                       '[H][C@@]1(CC(O2)=O)[C@@]32C[C@@]([C@]4([H])[C@H](OC)C[C@@]3([H])C(C)(C)O1)(O5)CC[C@@]6(C)[C@@]([C@]7([H])[C@H](C)C6=O)([H])[C@@]5(O[C@@]([C@@]8(C)O)([H])[C@]7([H])OC8=O)C4=O',
                       '[H][C@@]1(C2)[C@@]3(OC2=O)C[C@]4(O)[C@]([H])([C@@]5(O[C@H]5C[C@@]6([C@H](C)C7)[H])[C@]6(O[C@@]87OC(C(C)=C8)=O)[C@@H](O)C4)[C@H](OC(C)=O)C[C@@]3([H])C(C)(C)O1', 
                       'C[C@@H]1[C@H]2[C@H]3[C@H]([C@](C(=O)O3)(C)O)O[C@]45[C@H]2[C@](C1=O)(CC[C@@]6(O4)C[C@]78[C@H](C[C@@H]9[C@@]6(C5=O)O9)C(O[C@H]7CC(=O)O8)(C)C)C ']
        # test_smiles = test_df.SMILES
        predictions = classifier.predict_with_best_model(test_smiles)
        print("新分子预测结果 (使用最佳模型):")
        print(predictions)
        
        # 保存比较结果
        comparison_df.to_csv('model_comparison_results.csv', index=False)
        print(f"\n模型比较结果已保存到: model_comparison_results.csv")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()