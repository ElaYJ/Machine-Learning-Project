
import numpy as np
import pandas as pd


# 데이터 전처리
from sklearn.preprocessing import LabelEncoder


def get_outlier(df=None, column=None, weight=1.5):
    fraud = df[df['re_order']==0][column]
    quantile_25 = np.percentile(fraud.values, 25)
    quantile_75 = np.percentile(fraud.values, 75)
    
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight
    
    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index.to_list()
    
    return outlier_index



def label_encoding(df, cols):
    le = LabelEncoder()
    for col in cols:
        le.fit(df[col])
        rlt_df = pd.DataFrame(data=[le.classes_,le.transform(le.classes_)], index=('class', 'encoding'))
        display(rlt_df)
        df[f"le_{col}"] = le.transform(df[col])




# 베스트 모델 --> 하이퍼파라미터 튜닝 : GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV

def grid_search_cv_ressult(gs, cols=['params','rank_test_score','mean_train_score','mean_test_score']):
	score_df = pd.DataFrame(gs.cv_results_).sort_values(by='rank_test_score').reset_index(drop=True)
	return score_df[cols]


def get_gscv_best_model(X, y, clf=None, params=None, scoring='f1_macro', report=True, cols=None):
    skfold = StratifiedKFold(n_splits=5, shuffle=True)
    grid_cv = GridSearchCV(
		estimator=clf, param_grid=params, cv=skfold, scoring=scoring, 
  		return_train_score=True, n_jobs=-1
	)
    grid_cv.fit(X, y)
    print('Best Params:', grid_cv.best_params_)
    
    if report:
        if cols == None:
            cols=['params','rank_test_score','mean_train_score','mean_test_score']
        display(grid_search_cv_ressult(grid_cv, cols).head(7))
    
    return grid_cv.best_estimator_




# 성능 지표
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


# One Model ---------------------------------------------------------------------------

def print_clf_scores(y_test, pred_test):
	# print(' Accuracy :', accuracy_score(y_test, pred_test))
	# print(' Percision:', precision_score(y_test, pred_test))
	# print(' Recall:', recall_score(y_test, pred_test))
	# print(' F1 Score:', f1_score(y_test, pred_test))
	# print(' AUC Score:', roc_auc_score(y_test, pred_test))
	
	print(' Accuracy: {0:.4f},  Precision: {1:.4f}'.format(
		accuracy_score(y_test, pred_test), precision_score(y_test, pred_test)
	))
	print(' Recall: {0:.4f},  F1-score: {1:.4f},  AUC: {2:.4f}'.format(
		recall_score(y_test, pred_test), f1_score(y_test, pred_test), roc_auc_score(y_test, pred_test)
	))
 
 

def show_classification_report(y_test=None, pred_test=None, confusion=False):
    print(
		classification_report(y_test, pred_test, target_names=['Once Order(0)', 'Re Order(1)']),
        end='\n'
	)
    
    if confusion:
        print('《confusion matrix》')
        print(confusion_matrix(y_test, pred_test))
        print('='*57)
        print_clf_scores(y_test, pred_test)



def show_roc_curve(model_clf, X_test, y_test, show_rlt=False):
	pred_proba = model_clf.predict_proba(X_test)

	fpr, tpr, thresholds = roc_curve(y_test, pred_proba[:,1])

	if show_rlt:
		df = pd.DataFrame(
			data = [thresholds, fpr, tpr],
			index=['THs', 'FPR', 'TPR']
		)
		display(df)

	plt.figure(figsize=(7,6))
	plt.plot([0,1], [0,1], 'c', ls='dashed') #--> y=x 직선
	plt.plot(fpr, tpr, 'r') #--> x: fpr, y: tpr
	plt.grid()
	plt.show()



# Models ------------------------------------------------------------------------------

def get_clf_scores(y_test, pred):
    acc = accuracy_score(y_test, pred)
    precis = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    
    return acc, precis, recall, f1, auc


def get_result(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    return get_clf_scores(y_test, pred)


def get_results_df(models, X_train, y_train, X_test, y_test):
    col_names = ['accuracy', 'precision', 'recall', 'F1', 'roc_auc']
    
    results = []
    model_names = []
    for name, model in models:
        results.append(get_result(model, X_train, y_train, X_test, y_test))
        model_names.append(name)
    
    return pd.DataFrame(results, columns=col_names, index=model_names)


def show_models_roc(models, X_test, y_test, score=False):
	plt.figure(figsize=(10,8))
	plt.plot([0,1], [0,1], 'c', ls='--', label='random_guess')

	for model_name, model in models.items():
		pred = model.predict_proba(X_test)[:, 1]

		fpr, tpr, thresholds = roc_curve(y_test, pred)
		if score:
			print("==", model_name)
			df = pd.DataFrame(
				data = [thresholds, fpr, tpr],
				index = ['THs', 'FPR', 'TPR']
			)
			display(df)

		plt.plot(fpr, tpr, label=model_name)

	plt.legend(fontsize=12)
	plt.xlabel("FPR", fontsize=13)
	plt.ylabel("TPR", fontsize=13)
	plt.title("Models ROC Curve Comparision", fontsize=17)
	plt.grid()
	plt.show()



# Features Importance/Coefficient ------------------------------------------------------

def show_feature_importance(clf, labels=None):
    import_val = clf.feature_importances_
    df = pd.DataFrame({'Features':labels,'importance':import_val})
    df = df.sort_values(by='importance', ascending=False)
    display(df)
    
    df.set_index('Features', inplace=True)
    
    sns.barplot(x=df['importance'], y=df.index, hue=df.index, legend=False, palette='Spectral')
    plt.xlabel("Importance", fontsize=12)
    plt.title("Features Coef. Importance")
    plt.show()


def show_reg_feature_importance(coefs=None, labels=None):
    df = pd.DataFrame({'Features':labels,'importance':coefs})
    df = df.sort_values(by='importance')
    display(df)
    
    df['positive'] = df['importance'] > 0
    df.set_index('Features', inplace=True)
    
    df['importance'].plot(kind='barh', color=df.positive.map({True:'blue', False:'red'}))
    plt.xlabel("Importance", fontsize=12)
    plt.title("Features Coef. Importance")
    plt.show()