
# 축 서식 지정
def format_spines(ax, right_ax=False, top_ax=False):
    # Setting up colors
    ax.spines['bottom'].set_color('#CCCCCC') #--> 라이트 그레이(Light Gray)
    ax.spines['left'].set_color('#CCCCCC')
    if top_ax:
        ax.spines['top'].set_visible(False) #--> top 축은 그리지 않는다.
    else:
        ax.spines['top'].set_color('#CCCCCC')
    if right_ax:
        ax.spines['right'].set_color('#CCCCCC')
    else:
        ax.spines['right'].set_color('#FFFFFF') #--> 흰색
    ax.patch.set_facecolor('#FFFFFF')


def text_annotate(ax, with_p=False, ncount=None):
    for p in ax.patches:        
        if with_p:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate(
				text='{}\n{:.1f}%'.format(int(y), 100. * y / ncount), xy=(x.mean(), y),
				ha='center', va='bottom', fontsize=11, color='darkblue'
			)
        else:
            ax.annotate(
				text=f"{int(p.get_height())}", xy=(p.get_x()+p.get_width()/2, p.get_height()), xytext=(0,7),
				ha='center', va='center', textcoords='offset points', fontsize=11, color='darkblue'
			)



def set_axis(ax, title='', xlabel='', ylabel='', angle=None, pad=20, right_ax=False, top_ax=False):
    format_spines(ax, right_ax, top_ax)
    if angle:
        for tick in ax.get_xticklabels():
            tick.set(fontsize=12)
            tick.set_rotation(angle)
    else:
        for tick in ax.get_xticklabels():
            tick.set(fontsize=12)
    if len(xlabel):   ax.set_xlabel(xlabel, fontsize=13)
    if len(ylabel):   ax.set_ylabel(ylabel, fontsize=13)
    if len(title):    ax.set_title(title, size=17, color='dimgrey', pad=pad)

