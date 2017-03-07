// == mojo ====================================================================
//
//    Copyright (c) gnawice@gnawice.com. All rights reserved.
//	  See LICENSE in root folder
//
//    Permission is hereby granted, free of charge, to any person obtaining a
//    copy of this software and associated documentation files(the "Software"),
//    to deal in the Software without restriction, including without 
//    limitation the rights to use, copy, modify, merge, publish, distribute,
//    sublicense, and/or sell copies of the Software, and to permit persons to
//    whom the Software is furnished to do so, subject to the following 
//    conditions :
//
//    The above copyright notice and this permission notice shall be included
//    in all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
//    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
//    OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
//    THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// ============================================================================
//    layer.h:  defines layers for neural network
// ==================================================================== mojo ==


#pragma once

#include <string>
#include <sstream>

#include "core_math.h"
#include "activation.h"

namespace mojo
{

#include <windows.h>
	/*
	double PCFreq = 0.0;
	__int64 CounterStart = 0;

	void StartCounter()
	{
		LARGE_INTEGER li;
		if (!QueryPerformanceFrequency(&li)) return;
		PCFreq = double(li.QuadPart) / 1000.0;
		QueryPerformanceCounter(&li);
		CounterStart = li.QuadPart;
	}
	double GetCounter()
	{
		LARGE_INTEGER li;
		QueryPerformanceCounter(&li);
		return double(li.QuadPart - CounterStart) / PCFreq;
	}
	*/

#define int2str(a) std::to_string((long long)a)
#define float2str(a) std::to_string((long double)a)
#define bail(txt) {std::cerr << txt; throw;}

//----------------------------------------------------------------------------------------------------------
// B A S E   L A Y E R
//
// all other layers derived from this
class base_layer 
{
protected:
	bool _has_weights;
	bool _use_bias;
	float _learning_factor;
	int _thread_count;

public:
	activation_function *p_act;
	
	bool has_weights() {return _has_weights;}
	bool use_bias() { return _use_bias; }
	void set_learning_factor(float f=1.0f) {_learning_factor = 1.f;}
	void set_threading(int thread_count) {_thread_count=thread_count; if(_thread_count<1) _thread_count=1;}

	int pad_cols, pad_rows;
	matrix node;
	matrix bias; // this is something that maybe should be in the same class as the weights... but whatever. handled differently for different layers
	
	std::string name;
	// index of W matrix, index of connected layer
	std::vector<std::pair<int,base_layer*>> forward_linked_layers;
#ifndef MOJO_NO_TRAINING
	matrix delta;
	std::vector<std::pair<int,base_layer*>> backward_linked_layers;

	virtual void distribute_delta(base_layer &top, const matrix &w, const int train = 1) =0;
	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train =1)=0;
	virtual void update_bias(const matrix &newbias, float alpha) {};

#endif
	virtual void accumulate_signal(const base_layer &top_node, const matrix &w, const int train =0) =0;

	base_layer(const char* layer_name, int _w, int _h=1, int _c=1) : node(_w, _h, _c),  p_act(NULL), name(layer_name), _has_weights(true), pad_cols(0), pad_rows(0), _learning_factor(1.f), _use_bias(false), _thread_count(1)
		#ifndef MOJO_NO_TRAINING
		,delta(_w,_h,_c,NULL,false)
		#endif
	{
	}

	virtual void resize(int _w, int _h=1, int _c=1)
	{
		if (_w<1) _w = 1; if (_h<1) _h = 1; if (_c<1) _c = 1;
		node =matrix(_w,_h,_c);
		if (_use_bias) { bias = matrix(_w, _h, _c); bias.fill(0.); }
		#ifndef MOJO_NO_TRAINING
		delta =matrix(_w,_h,_c,NULL,false);
		#endif
	}
	
	virtual ~base_layer(){if(p_act) delete p_act;}
	virtual int fan_size() {return node.chans*node.rows*node.cols;}

	virtual void activate_nodes()
	{
		if (p_act)
		{
			if(_use_bias)
				//for (int c=0; c<node.chans; c++) 
				{
					//const float b = bias.x[c];
					//float *x= &node.x[c*node.chan_stride];
					p_act->f(node.x,node.size(),bias.x);
				}
			else
				p_act->f(node.x, node.size(), 0);

		}
	}
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
		top.forward_linked_layers.push_back(std::make_pair((int)weight_mat_index,this));
		#ifndef MOJO_NO_TRAINING
		backward_linked_layers.push_back(std::make_pair((int)weight_mat_index,&top));
		#endif
		if (_has_weights)
		{
			int rows = node.cols*node.rows*node.chans;
			int cols = top.node.cols*top.node.rows*top.node.chans;
			return new matrix(cols, rows, 1);
		}
		else
			return NULL;	
	}

	//inline float f(float *in, int i, int size, float bias) {return p_act->f(in, i, size, bias);};
	inline float df(float *in, int i, int size) { if (p_act) return p_act->df(in, i, size); else return 1.f; };
	virtual std::string get_config_string() =0;	
};

//----------------------------------------------------------------------------------------------------------
// I N P U T   L A Y E R
//
// input layer class - can be 1D, 2D (c=1), or stacked 2D (c>1)
class input_layer : public base_layer
{
public:
	input_layer(const char *layer_name, int _w, int _h=1, int _c=1) : base_layer(layer_name,_w,_h,_c) {p_act=new_activation_function("identity"); }
	virtual  ~input_layer(){}
	virtual void activate_nodes() { /*node.reset_empty_chans(); */}
	virtual void distribute_delta(base_layer &top, const matrix &w, const int train =1) {}
	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train =1) {}
	virtual void accumulate_signal(const base_layer &top_node, const matrix &w, const int train =0) {}
	virtual std::string get_config_string() {std::string str="input "+int2str(node.cols)+" "+int2str(node.rows)+" "+int2str(node.chans)+ " "+p_act->name+"\n"; return str;}
};

//----------------------------------------------------------------------------------------------------------
// F U L L Y   C O N N E C T E D
//
// fully connected layer
class fully_connected_layer : public base_layer
{
public:
	fully_connected_layer(const char *layer_name, int _size, activation_function *p) : base_layer(layer_name, _size, 1, 1) 
	{
		p_act = p; _use_bias = true;	
		bias = matrix(node.cols, node.rows, node.chans);
		bias.fill(0.);

	}//layer_type=fully_connected_type;}
	virtual std::string get_config_string() {std::string str="fully_connected "+int2str(node.size())+ " "+p_act->name+"\n"; return str;}
	virtual void accumulate_signal( const base_layer &top,const matrix &w, const int train =0)
	{
		// doesn't care if shape is not 1D
		// here weights are formated in matrix, top node in cols, bottom node along rows. (note that my top is opposite of traditional understanding)
		// node += top.node.dot_1dx2d(w);
		const int s = w.rows;
		const int ts = top.node.size();
		const int ts2 = top.node.cols*top.node.rows;

		// this can be sped up a little with SSE. 
		if(top.node.chan_stride!=ts2)
		{
			//std::cout << "here: " << top.node.chan_stride << ","<< ts2 << ","<< top.node.chans << ":";
			MOJO_THREAD_THIS_LOOP(_thread_count)
			for (int j = 0; j < s; j++)  
			{
				for (int i = 0; i < top.node.chans; i++) 
				{
					node.x[j] += dot(top.node.x+top.node.chan_stride*i, w.x+j*w.cols+ts2*i, ts2);  
					//float *f=top.node.x+top.node.chan_stride*i;
					//if(node.x[j]!=node.x[j])
					if(node.x[j]!=node.x[j])
					{
						//std::cout << "stuff" << top.name << " " << name << " " << top.node.x[top.node.chan_stride*i] << " " << w.x[j*w.cols+ts2*i] << " | " ;
						for (int k=0; k<top.node.size(); k++)
						{
							std::cout << k<< ","<< top.node.x[k] <<",";
						}
						
						exit(1);
					}
				}
			}
		}
		else
		{
		MOJO_THREAD_THIS_LOOP(_thread_count)
		for (int j = 0; j < s; j++)  node.x[j] += dot(top.node.x, w.x+j*w.cols, ts);  
		}
	}
#ifndef MOJO_NO_TRAINING
	virtual void update_bias(const matrix &newbias, float alpha) {
		for (int j = 0; j < bias.size(); j++) bias.x[j] -= newbias.x[j] * alpha;
	}
	virtual void distribute_delta(base_layer &top, const matrix &w, const int train =1)
	{
		if(top.delta.cols*top.delta.rows==top.delta.chan_stride)
		{
			const int w_cols = w.cols;
			for (int b = 0; b < delta.size(); b++)
			{
				const float cb = delta.x[b];
				for (int t = 0; t < top.delta.size(); t++) 
					top.delta.x[t] += cb*w.x[t + b*w_cols];
			}
		}
		else
		{
			const int w_cols = w.cols;
			const int chan_size=top.delta.cols*top.delta.rows;

			for (int b = 0; b < delta.size(); b++)
			{
				const float cb = delta.x[b];
				for (int tc = 0; tc < top.delta.chans; tc++)	
					for (int t = 0; t < chan_size; t++)	
							top.delta.x[t+tc*top.delta.chan_stride] += cb*w.x[t + tc*chan_size + b*w_cols];
			}

				 
			
		}


	}

	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train = 1)
	{
		const float *bottom = delta.x; const int sizeb = delta.size();
		const float *top = top_layer.node.x; const int sizet = top_layer.node.cols*top_layer.node.rows*top_layer.node.chans;
		dw.resize(sizet, sizeb, 1);

		for (int b = 0; b < sizeb; b++)
		{
			const float cb = bottom[b];
			const int chan_size =  top_layer.node.cols*top_layer.node.rows;
			if(sizet!=top_layer.node.size())
			{
				//std::cout << "calculate_dw - odd size";
				for (int tc = 0; tc < top_layer.node.chans; tc++)	
					for (int t = 0; t < chan_size; t++)	
					{
						dw.x[t+tc*chan_size + b*sizet] = top[t+tc*top_layer.node.chan_stride] * cb;
						//std::cout << dw.x[t+tc*chan_size + b*sizet] <<",";
					}
				 
			}
			else
			{
				for (int t = 0; t < sizet; t++)	dw.x[t + b*sizet] = top[t] * cb;
			}
		}
	}
#endif

};

//----------------------------------------------------------------------------------------------------------
// M A X   P O O L I N G   
// 
// may split to max and ave pool class derived from pooling layer.. but i never use ave pool anymore
class max_pooling_layer : public base_layer
{

protected:
	int _pool_size;
	int _stride;
	// uses a map to connect pooled result to top layer
	std::vector<int> _max_map;
public:
	max_pooling_layer(const char *layer_name, int pool_size) : base_layer(layer_name, 1)
	{
		 _stride = pool_size; _pool_size = pool_size; //layer_type=pool_type;
		_has_weights = false;
	}
	max_pooling_layer(const char *layer_name, int pool_size, int stride ) : base_layer(layer_name, 1)
	{
		_stride= stride; _pool_size=pool_size;  //layer_type=pool_type;
		_has_weights = false;
	}
	virtual  ~max_pooling_layer(){}
	virtual std::string get_config_string() {std::string str="max_pool "+int2str(_pool_size) +" "+ int2str(_stride) +"\n"; return str;}

	// ToDo would like delayed activation of conv layer if available
//	virtual void activate_nodes(){ return;}
	virtual void resize(int _w, int _h=1, int _c=1)
	{
		if(_w<1) _w=1; if(_h<1) _h=1; if(_c<1) _c=1;
		_max_map.resize(_w*_h*_c);
		base_layer::resize(_w, _h, _c);
	}
	// no weights 
	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train =1) {}
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
			// need to set the size of this layer
		// can really only handle one connection comming in to this
		int pool_size = _pool_size;
		int w = (top.node.cols) / pool_size;
		int h = (top.node.rows) / pool_size;
		if (_stride != _pool_size)
		{
			w = 1 + ((top.node.cols - _pool_size) / _stride);
			h = 1 + ((top.node.rows - _pool_size) / _stride);
		}

		resize(w, h, top.node.chans);

		return base_layer::new_connection(top, weight_mat_index);
	}

	// this is downsampling
	// the pool size must fit correctly in the image map (use resize prior to call if this isn't the case)
	virtual void accumulate_signal(const base_layer &top,const matrix &w,const int train =0)
	{
		int kstep = top.node.chan_stride; // top.node.cols*top.node.rows;
		int jstep=top.node.cols;
		int output_index=0;
		int *p_map = _max_map.data();
		int pool_y=_pool_size; if(top.node.rows==1) pool_y=1; //-top.pad_rows*2==1) pool_y=1;
		int pool_x=_pool_size; if(top.node.cols==1) pool_x=1;//-top.pad_cols*2==1) pool_x=1;
		const float *top_node = top.node.x;

		for(int k=0; k<top.node.chans; k++)
		{
			for(int j=0; j<=top.node.rows- _pool_size; j+= _stride)
			{
				for(int i=0; i<=top.node.cols- _pool_size; i+= _stride)
				{
					const int base_index=i+(j)*jstep+k*kstep;
					int max_i=base_index;
					float max=top_node[base_index];
					if(pool_x==2)
					{
						const float *n=top_node+base_index;
						//if(max<n[0]) { max = n[0]; max_i=max_i;}
						if(max<n[1]) { max = n[1]; max_i=base_index+1;}
						n+=jstep;
						if(max<n[0]) { max = n[0]; max_i=base_index+jstep;}
						if(max<n[1]) { max = n[1]; max_i=base_index+jstep+1;}
					}
					else if(pool_x==3)
					{
						const float *n=top_node+base_index;
						//if(max<n[0]) { max = n[0]; max_i=max_i;}
						if(max<n[1]) { max = n[1]; max_i=base_index+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+2;}
						n+=jstep;
						if(max<n[0]) { max = n[0]; max_i=base_index+jstep;}
						if(max<n[1]) { max = n[1]; max_i=base_index+jstep+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+jstep+2;}
						n+=jstep;
						if(max<n[0]) { max = n[0]; max_i=base_index+2*jstep;}
						if(max<n[1]) { max = n[1]; max_i=base_index+2*jstep+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+2*jstep+2;}
					}
					else if(pool_x==4)
					{
						const float *n=top_node+base_index;
						//if(max<n[0]) { max = n[0]; max_i=max_i;}
						if(max<n[1]) { max = n[1]; max_i=base_index+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+2;}
						if(max<n[3]) { max = n[3]; max_i=base_index+3;}
						n+=jstep;
						if(max<n[0]) { max = n[0]; max_i=base_index+jstep;}
						if(max<n[1]) { max = n[1]; max_i=base_index+jstep+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+jstep+2;}
						if(max<n[3]) { max = n[3]; max_i=base_index+jstep+3;}
						n+=jstep;
						if(max<n[0]) { max = n[0]; max_i=base_index+2*jstep;}
						if(max<n[1]) { max = n[1]; max_i=base_index+2*jstep+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+2*jstep+2;}
						if(max<n[3]) { max = n[3]; max_i=base_index+2*jstep+3;}
						n+=jstep;
						if(max<n[0]) { max = n[0]; max_i=base_index+3*jstep;}
						if(max<n[1]) { max = n[1]; max_i=base_index+3*jstep+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+3*jstep+2;}
						if(max<n[3]) { max = n[3]; max_i=base_index+3*jstep+3;}
					}
					else	
					{
						// speed up with optimized size version
						for(int jj=0; jj<pool_y; jj+= 1)
						{
							for(int ii=0; ii<pool_x; ii+= 1)
							{
								int index=i+ii+(j+jj)*jstep+k*kstep;
								if((max)<(top_node[index]))
								{
									max = top_node[index];
									max_i=index;

								}
							}
						}

					}
					//if (max<1e-5) node.empty_chan[k] = 1;
					//else node.empty_chan[k] = 0;

					node.x[output_index] = top_node[max_i];
					p_map[output_index] = max_i;
					output_index++;
					
				}
			}
		}
	}
#ifndef MOJO_NO_TRAINING

	// this is upsampling
	virtual void distribute_delta(base_layer &top, const matrix &w, const int train =1)
	{
		int *p_map = _max_map.data();
		const int s = (int)_max_map.size();
		for(int k=0; k<s; k++) top.delta.x[p_map[k]]+=delta.x[k];
	}
#endif
};

//----------------------------------------------------------------------------------------------------------
// S E M I   S T O C H A S T I C   P O O L I N G   
// concept similar to stochastic pooling but only slects 'max' based on top 2 candidates
class semi_stochastic_pooling_layer :  public max_pooling_layer
{
public:
	semi_stochastic_pooling_layer(const char *layer_name, int pool_size) : max_pooling_layer(layer_name, pool_size) {}
	
	semi_stochastic_pooling_layer(const char *layer_name, int pool_size, int stride) : max_pooling_layer(layer_name, pool_size, stride){}

	virtual std::string get_config_string() { std::string str = "semi_stochastic_pool " + int2str(_pool_size) + " " + int2str(_stride) + "\n"; return str; }
	virtual void accumulate_signal(const base_layer &top, const matrix &w, const int train = 0)
	{
		int kstep = top.node.cols*top.node.rows;
		int jstep = top.node.cols;
		int output_index = 0;
		int *p_map = _max_map.data();
		int pool_y = _pool_size; if (top.node.rows == 1) pool_y = 1; //-top.pad_rows*2==1) pool_y=1;
		int pool_x = _pool_size; if (top.node.cols == 1) pool_x = 1;//-top.pad_cols*2==1) pool_x=1;
		const float *top_node = top.node.x;
		for (int k = 0; k<top.node.chans; k++)
		{

			for (int j = 0; j <= top.node.rows - _pool_size; j += _stride)
			{
				for (int i = 0; i <= top.node.cols - _pool_size; i += _stride)
				{
					const int base_index = i + (j)*jstep + k*kstep;
					int max_i = base_index;
					float max = top_node[base_index];
					int max2_i = base_index;
					float max2 = max;
					// speed up with optimized size version
					for (int jj = 0; jj < pool_y; jj += 1)
					{
						for (int ii = 0; ii < pool_x; ii += 1)
						{
							int index = i + ii + (j + jj)*jstep + k*kstep;
							if ((max) < (top_node[index]))
							{
								max2 = max;
								max2_i = max_i;

								max = top_node[index];
								max_i = index;

							}
							else if ((max2) < (top_node[index]))
							{
								max2 = top_node[index];
								max2_i = index;
							}
						}
					}

//					if(max<1e-5) node.empty_chan[k] = 1;
	//				else node.empty_chan[k] = 0;

					int r = rand() % 100;
					float denom = (max + max2);
					
					if (denom == 0)
					{
						node.x[output_index] = top_node[max_i];
						p_map[output_index] = max_i;
					}
					else
					{
						int t1 = (int)(100 * max / (max + max2));
						if (r <= t1 || train == 0)
						{
							node.x[output_index] = top_node[max_i];
							p_map[output_index] = max_i;
						}
						else
						{
							node.x[output_index] = top_node[max2_i];
							p_map[output_index] = max2_i;
						}
					}
					output_index++;
				}
			}
		}
	}

};

//----------------------------------------------------------------------------------------------------------
// D R O P   O U T
// 
class dropout_layer : public base_layer
{
	float _dropout_rate;
	//std::map<const base_layer*, matrix>  drop_mask;
	matrix  drop_mask;
public:
	dropout_layer(const char *layer_name, float dropout_rate) : base_layer(layer_name, 1)
	{
		_has_weights = false;
		_dropout_rate = dropout_rate;
		p_act = NULL;// new_activation_function("identity");
	}
	virtual  ~dropout_layer() {}
	virtual std::string get_config_string() { std::string str = "dropout " + float2str(_dropout_rate)+"\n"; return str; }
	virtual void resize(int _w, int _h = 1, int _c = 1)
	{
		if (_w<1) _w = 1; if (_h<1) _h = 1; if (_c<1) _c = 1;
		drop_mask.resize(_w, _h, _c);
		base_layer::resize(_w, _h, _c);
	}

	// no weights 
	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train = 1) {}
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
		resize(top.node.cols, top.node.rows, top.node.chans);
		return base_layer::new_connection(top, weight_mat_index);
	}

	// for dropout...
	// we know this is called first in the backward pass, and the train will be set to 1
	// when that happens the dropouts will be set. 
	// different dropouts for each mininbatch... don't know if that matters...
	virtual void accumulate_signal(const base_layer &top, const matrix &w, const int train = 0)
	{
		const float *top_node = top.node.x;
		const int size = top.node.chans*top.node.rows*top.node.cols;
		memcpy(node.x, top_node, sizeof(float)*size);
//		matrix *pmask = &(drop_mask[&top]);
		matrix *pmask = &drop_mask;
		if (train)
		{
			pmask->fill(1);
			int k;
			for (k = 0; k < size; k+=4) // do 4 at a time
			{
				int r = rand();
				if ((r % 100) <= (_dropout_rate*100.f)) { pmask->x[k] = 0.0;  node.x[k] *= 0.5f; };
				if (((r >> 1) % 100) <= (_dropout_rate*100.f)) { pmask->x[k + 1] = 0.0; node.x[k + 1] *= 0.5f; }
				if (((r >> 2) % 100) <= (_dropout_rate*100.f)) { pmask->x[k + 2] = 0.0;  node.x[k + 2] *= 0.5f; }
				if (((r >> 3) % 100) <= (_dropout_rate*100.f)) { pmask->x[k + 3] = 0.0;  node.x[k + 3] *= 0.5f; }
			}
			int k2 = k - 4;
			for (k = k2; k < size; k++)
			{
				int r = rand();
				if ((r % 100) <= (_dropout_rate*100.f)) { pmask->x[k] = 0.0;  node.x[k] *= 0.5f; };
			}
		}
	}
#ifndef MOJO_NO_TRAINING

	virtual void distribute_delta(base_layer &top, const matrix &w, const int train = 1)
	{
//		delta *= drop_mask[&top];
		delta *= drop_mask;
		top.delta += delta;
	}
#endif
};

//----------------------------------------------------------------------------------------------------------
// M F M   -  M a x F e a t u r e M a p 
// (A Lightened CNN for Deep Face Representation) http://arxiv.org/pdf/1511.02683.pdf
// the parameter passed in is the number of maps pooled
class maxout_layer : public base_layer
{
	int _pool;
	matrix max_map;
public:
	maxout_layer(const char *layer_name, int  pool_chans) : base_layer(layer_name, 1)
	{
		_pool = pool_chans;
		if (_pool < 2) _pool = 2;
		p_act = new_activation_function("identity");
		_has_weights = false;
	}
	virtual  ~maxout_layer() {}
	virtual std::string get_config_string() { std::string str = "mfm" + int2str(_pool) + "\n"; return str; }
	virtual void resize(int _w, int _h = 1, int _c = 1)
	{
		_c /= _pool;
		if (_w<1) _w = 1; if (_h<1) _h = 1; if (_c<1) _c = 1;
		max_map.resize(_w, _h, _c);
		base_layer::resize(_w, _h, _c);
	}

	inline float df(float *in, int i, int size) 
	{
		return 1.;
	};

	virtual void activate_nodes() { return; }
	// no weights 
	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train = 1) {}
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
		// wasteful to add weight matrix (1x1x1), but makes other parts of code more OO
		// bad will happen if try to put more than one pool layer
		top.forward_linked_layers.push_back(std::make_pair(weight_mat_index, this));
		int w = (top.node.cols) / 1;
		int h = (top.node.rows) / 1;
		resize(w, h, top.node.chans);
#ifndef MOJO_NO_TRAINING
		backward_linked_layers.push_back(std::make_pair(weight_mat_index, &top));
#endif
		return NULL;
		//return new matrix(1, 1, 1);
	}

	// for maxout
	// we know this is called first in the backward pass, and the train will be set to 1
	// when that happens the dropouts will be set. 
	// different dropouts for each mininbatch... don't know if that matters...
	virtual void accumulate_signal(const base_layer &top, const matrix &w, const int train = 0)
	{
		const float *top_node = top.node.x;
		const int chan_size = top.node.rows*top.node.cols;
		//const int pool_offset = top.node.chans / _pool;
		const int s = chan_size*top.node.chans / _pool;

		if((top.node.chans % _pool) !=0) 
			bail("mfm layer has pool size that is not a multiple of the input channels");
		
		for (int i = 0; i < s; i++)
				{
			float max = top.node.x[i];
			int maxk = i;
					for (int k = 1; k < _pool; k++)
					{
				if (top.node.x[i + (k*s)] > max)
						{
							//node.x[i + c / 2 * chan_size] = max;
					max = top.node.x[i + (k*s)];
					maxk = i + (k*s);
							// max_map tells which map 0 or 1 when pooling
							//max_map.x[i + c / 2 * chan_size] = 0;
						}
					}
			node.x[i] = max;
			max_map.x[i] = (float)maxk;
				}
			}
#ifndef MOJO_NO_TRAINING

	virtual void distribute_delta(base_layer &top, const matrix &w, const int train = 1)
	{
	//	const int chan_size = node.cols*node.rows;
	//	const int pool_offset = top.node.chans / _pool;
		const int chan_size = top.node.rows*top.node.cols;
		//const int pool_offset = top.node.chans / _pool;
		const int s = chan_size*top.node.chans / _pool;
		for (int c = 0; c < s; c++)
			{
//			for (int k = 0; k < node.cols*node.rows; k++)
//			{
				int maxmap = (int)max_map.x[c];
				top.delta.x[maxmap] += delta.x[c];
//			}
		}

	}
#endif
};

//----------------------------------------------------------------------------------------------------------
// C O N V O L U T I O N   
//
class convolution_layer : public base_layer
{
	int _stride;
public:
	int kernel_rows;
	int kernel_cols;
	int maps;
	//int maps_per_kernel;
	int kernels_per_map;


	convolution_layer(const char *layer_name, int _w, int _c, int _s, activation_function *p ) : base_layer(layer_name, _w, _w, _c) 
	{
		p_act=p; _stride =_s; kernel_rows=_w; kernel_cols=_w; maps=_c;kernels_per_map=0; pad_cols = kernel_cols-1; pad_rows = kernel_rows-1;
		_use_bias = true;
	}
	virtual  ~convolution_layer() {
	}
	virtual std::string get_config_string() {std::string str="convolution "+int2str(kernel_cols)+" "+int2str(maps)+" " + int2str(_stride) + " " +p_act->name+"\n"; return str;}
	
	virtual int fan_size() { return kernel_rows*kernel_cols*maps *kernels_per_map; }

	
	virtual void resize(int _w, int _h=1, int _c=1) // special resize nodes because bias handled differently with shared wts
	{
		if(kernel_rows*kernel_cols==1) node =matrix(_w,_h,_c);  /// use special channel aligned matrix object
		else node =matrix(_w,_h,_c,NULL,true);  /// use special channel aligned matrix object

		bias =matrix(1,1,_c);
		bias.fill(0.);
		#ifndef MOJO_NO_TRAINING
		if(kernel_rows*kernel_cols==1) delta =matrix(_w,_h,_c);  /// use special channel aligned matrix object
		else delta =matrix(_w,_h,_c,NULL,true);  /// use special channel aligned matrix object
		#endif
	}

	// this connection work won't work with multiple top layers (yet)
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
		top.forward_linked_layers.push_back(std::make_pair(weight_mat_index,this));
		#ifndef MOJO_NO_TRAINING
		backward_linked_layers.push_back(std::make_pair(weight_mat_index,&top));
		#endif
		// re-shuffle these things so weights of size kernel w,h,kerns - node of size see below
		//int total_kernels=top.node.chans*node.chans;
		kernels_per_map += top.node.chans;
		resize((top.node.cols-kernel_cols)/_stride+1, (top.node.rows-kernel_rows)/_stride+1, maps);

		return new matrix(kernel_cols,kernel_rows, maps*kernels_per_map);
	}

	// activate_nodes
	virtual void activate_nodes()
	{ 
		const int map_size = node.rows*node.cols;
		const int map_stride = node.chan_stride;
		const int _maps = maps;

		MOJO_THREAD_THIS_LOOP(_thread_count)
		for (int c=0; c<_maps; c++) 
		{
			p_act->fc(&node.x[c*map_stride],map_size,bias.x[c]);
			//if(node.x[c*map_stride]!=node.x[c*map_stride]) bail("activate");
			
		}
	}


	virtual void accumulate_signal( const base_layer &top, const matrix &w, const int train =0)
	{	
		const int kstep = top.node.chan_stride;// NOT the same as top.node.cols*top.node.rows;
		const int jstep=top.node.cols;
		//int output_index=0;
		const int kernel_size=kernel_cols*kernel_rows;
		const int kernel_map_step = kernel_size*kernels_per_map;
		const int map_size=node.cols*node.rows;
		const int map_stride = node.chan_stride;

		const float *_w = w.x;
		const int top_chans = top.node.chans;
		const int map_cnt=maps;
		const int w_size = kernel_cols;
		const int stride = _stride;		
		const int node_size= node.cols;
		const int top_node_size = top.node.cols;
		const int outsize = node_size*node_size;

		if(kernel_rows>=2 && (kernel_rows<=5))
		{
			matrix img_ptr(node_size, node_size, kernel_rows*kernel_rows, NULL, true);
			for (int k = 0; k < top_chans; k++) // input channels --- same as kernels_per_map - kern for each input
			{
				unwrap_aligned_NxN(kernel_rows, img_ptr.x, &top.node.x[k*kstep], jstep, stride);
				float *ww = &w.x[(0 + k*maps)*kernel_size];

				if(kernel_rows==2)
				{
					MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
					for (int map = 0; map < map_cnt; map+=1) dotsum_unwrapped_2x2(img_ptr.x, ww+map*kernel_size, node.x + map_stride*map, outsize);
				}
				else if(kernel_rows==3)
				{
					MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
					for (int map = 0; map < map_cnt; map+=1) dotsum_unwrapped_3x3(img_ptr.x, ww+map*kernel_size, node.x + map_stride*map, outsize);
				}
				else if(kernel_rows==4)
				{
					MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
					for (int map = 0; map < map_cnt; map+=1) dotsum_unwrapped_4x4(img_ptr.x, ww+map*kernel_size, node.x + map_stride*map, outsize);
				}
				else //(kernel_rows==5)
				{
					MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
					for (int map = 0; map < map_cnt; map+=1) dotsum_unwrapped_5x5(img_ptr.x, ww+map*kernel_size, node.x + map_stride*map, outsize);
				}
			}
		}
		else if (kernel_rows == 1)
		{
			for (int k = 0; k < top_chans; k++) // input channels --- same as kernels_per_map - kern for each input
			{
				const float *_top_node = &top.node.x[k*kstep];

				//MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
				for (int map = 0; map < map_cnt; map++) 
				{
					const float cw = w.x[(map + k*maps)*kernel_size];
					const int mapoff = map_size*map;
					for (int j = 0; j < node_size*node_size; j += stride) node.x[j + mapoff] += _top_node[j] * cw;
				}
			}
		}
		else
		{
			for(int map=0; map<maps; map++) // how many maps  maps= node.chans
			{
				for(int k=0; k<top_chans; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
					for(int j=0; j<node_size; j+= stride) // input h 
 						for(int i=0; i<node_size; i+= stride) // intput w
							node.x[i+(j)*node.cols +map_stride*map]+= 
								unwrap_2d_dot(
									&top.node.x[(i)+(j)*jstep + k*kstep],
									&w.x[(map+k*maps)*kernel_size],
									kernel_cols,
									jstep,kernel_cols);
					
				} // k
			} // all maps=chans
		} 
			
	}


#ifndef MOJO_NO_TRAINING

	// convolution::distribute_delta
	virtual void distribute_delta(base_layer &top, const matrix &w, const int train=1)
	{
		
		// here to calculate top_delta += bottom_delta * W
//		top_delta.x[s] += bottom_delta.x[t]*w.x[s+t*w.cols];
		matrix delta_pad(delta, pad_cols, pad_rows);

		//const int kstep=top.delta.cols*top.delta.rows;
		const int kstep=top.delta.chan_stride;

		const int jstep=top.delta.cols;
		const int output_index=0;
		const int kernel_size=kernel_cols*kernel_rows;
		const int kernel_map_step = kernel_size*kernels_per_map;
		const int map_size=delta_pad.cols*delta_pad.rows;
		const int map_stride=delta_pad.chan_stride;

		const float *_w = w.x;
		const int w_size = kernel_cols;
		const int delta_size = delta_pad.cols;
		const int map_cnt=maps;
		const int top_delta_size = top.delta.rows;
		const int top_delta_chans = top.delta.chans;
		const int stride = _stride;

		matrix delt(top.delta.cols, top.delta.rows, top.delta.chans,NULL,true);
		
		if (kernel_cols == 5)
		{
			//*
					matrix img_ptr(delta_size, delta_size, 25, NULL, true);
					matrix filter_ptr(28, 1);
					
					//matrix imgout_ptr(outsize + 7, 1);
					for (int map = 0; map < map_cnt; map+=1) // how many maps  maps= node.chans
					{
						unwrap_aligned_NxN(5, img_ptr.x, &delta_pad.x[map*map_stride], delta_size, stride);
						const int outsize = top_delta_size*top_delta_size;
						for (int k = 0; k < top_delta_chans; k++) // input channels --- same as kernels_per_map - kern for each input
						{
							_w = &w.x[(k*maps + map)*kernel_size];
							// flip-flip to make 180 version
							for (int ii = 0; ii < 25; ii++) filter_ptr.x[ii] = _w[24 - ii];
							//float *out = node.x + map_stride*map;
							//float *out = &top.delta.x[k*kstep];
							float *out = &delt.x[k*delt.chan_stride];
							memcpy(out,&top.delta.x[k*kstep],sizeof(float)*outsize);
							dotsum_unwrapped_5x5(img_ptr.x, filter_ptr.x, out, outsize);// imgout_ptr.x, outsize);
							memcpy(&top.delta.x[k*kstep],out,sizeof(float)*outsize);
						}
					}
					/*/
			matrix filter_ptr(28, 1);
			matrix img_ptr(28 * delta_size*delta_size, 1);
			matrix imgout_ptr(delta_size*delta_size, 1);

			for (int map = 0; map < map_cnt; map++) // how many maps  maps= node.chans
			{
				unwrap_aligned_5x5(img_ptr.x, &delta_pad.x[map*map_stride], delta_size, stride);
				const int outsize = top_delta_size*top_delta_size;
				for (int k = 0; k < top_delta_chans; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_w = &w.x[(k*maps + map)*kernel_size];
					// flip-flip to make 180 version
					for (int ii = 0; ii < 25; ii++) filter_ptr.x[ii] = _w[24 - ii];

					dot_unwrapped_5x5(img_ptr.x, filter_ptr.x, imgout_ptr.x, outsize);

					float *out = &top.delta.x[k*kstep];
					for (int j = 0; j < outsize; j++) out[j] += imgout_ptr.x[j];

				}

			}
			//*/
		//	return;
		}
		else if(kernel_cols==3)
		{
					matrix img_ptr(delta_size, delta_size, 9, NULL, true);
					matrix filter_ptr(9, 1);
					
					//matrix imgout_ptr(outsize + 7, 1);
					for (int map = 0; map < map_cnt; map+=1) // how many maps  maps= node.chans
					{
						unwrap_aligned_NxN(3, img_ptr.x, &delta_pad.x[map*map_stride], delta_size, stride);
						const int outsize = top_delta_size*top_delta_size;
						for (int k = 0; k < top_delta_chans; k++) // input channels --- same as kernels_per_map - kern for each input
						{
							_w = &w.x[(k*maps + map)*kernel_size];
							// flip-flip to make 180 version
							for (int ii = 0; ii < 9; ii++) filter_ptr.x[ii] = _w[8 - ii];
							//float *out = node.x + map_stride*map;
						//	float *out = &top.delta.x[k*kstep];
						//	dotsum_unwrapped_3x3(img_ptr.x, filter_ptr.x, out, outsize);// imgout_ptr.x, outsize);
							float *out = &delt.x[k*delt.chan_stride];
							memcpy(out,&top.delta.x[k*kstep],sizeof(float)*outsize);
							dotsum_unwrapped_3x3(img_ptr.x, filter_ptr.x, out, outsize);// imgout_ptr.x, outsize);
							memcpy(&top.delta.x[k*kstep],out,sizeof(float)*outsize);
						}
					}
		}
		else if (kernel_cols == 2)
		{
			matrix img_ptr(delta_size, delta_size, 4, NULL, true);
			matrix filter_ptr(4, 1);
			matrix out_aligned(top_delta_size,top_delta_size,1,NULL,true);
			//matrix imgout_ptr(outsize + 7, 1);
			for (int map = 0; map < map_cnt; map+=1) // how many maps  maps= node.chans
			{
				unwrap_aligned_NxN(2, img_ptr.x, &delta_pad.x[map*map_stride], delta_size, stride);
				const int outsize = top_delta_size*top_delta_size;
				for (int k = 0; k < top_delta_chans; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_w = &w.x[(k*maps + map)*kernel_size];
					// flip-flip to make 180 version
					for (int ii = 0; ii < 4; ii++) filter_ptr.x[ii] = _w[3 - ii];
					memcpy(out_aligned.x, &top.delta.x[k*kstep],outsize*sizeof(float));
					//float *out = node.x + map_stride*map;
					float *out = out_aligned.x;// &top.delta.x[k*kstep];
					dotsum_unwrapped_2x2(img_ptr.x, filter_ptr.x, out, outsize);// imgout_ptr.x, outsize);
					memcpy(&top.delta.x[k*kstep],out_aligned.x,outsize*sizeof(float));

				}
			}
		}
		else if (kernel_cols == 1)
		{
			for (int j = 0; j<top.delta.rows; j += stride) // input h 
			{
				for (int i = 0; i<top.delta.cols; i += stride) // intput w
				{
					for (int k = 0; k<top.delta.chans; k++) // input channels --- same as kernels_per_map - kern for each input
					{
						int td_i = i + (j)*jstep + k*kstep;
						float *delt = &delta_pad.x[i + (j)*delta_pad.cols + 0*map_stride];
						float *wx = &w.x[(0 + k*maps)*kernel_size];
						for (int map = 0; map<maps; map++) // how many maps  maps= node.chans
						{
							top.delta.x[td_i] += (*delt)  * (*wx);
							delt += map_stride;
							wx += kernel_size;

						} // all input chans
						  //output_index++;	
					}
				}
			} //y
		}
		else
		{

			for(int j=0; j<top.delta.rows; j+=stride) // input h 
			{
				for(int i=0; i<top.delta.cols; i+=stride) // intput w
				{
					for(int k=0; k<top.delta.chans; k++) // input channels --- same as kernels_per_map - kern for each input
					{
						int td_i = i+(j)*jstep + k*kstep;
						for(int map=0; map<maps; map++) // how many maps  maps= node.chans
						{
							top.delta.x[td_i] += unwrap_2d_dot_rot180(
								&delta_pad.x[i+(j)*delta_pad.cols + map*map_stride], 
								&w.x[(map+k*maps)*kernel_size],
								kernel_cols,
								delta_pad.cols,kernel_cols);

						} // all input chans
						//output_index++;	
					} 
				}
			} //y
	
		} // all maps=chans 

	}


	// convolution::calculate_dw
	virtual void calculate_dw(const base_layer &top, matrix &dw, const int train =1)
	{
		int kstep=top.delta.chan_stride;
		int jstep=top.delta.cols;
		int output_index=0;
		int kernel_size=kernel_cols*kernel_rows;
		int kernel_map_step = kernel_size*kernels_per_map;
		int map_size=delta.cols*delta.rows;
		int map_stride=delta.chan_stride;

		dw.resize(kernel_cols, kernel_rows,kernels_per_map*maps);
		dw.fill(0);
		
		// node x already init to 0
		output_index=0;
		const int stride = _stride;
		const int top_node_size= top.node.cols;
		const int node_size = node.rows;
		const int delta_size = delta.cols;
		const int kern_len=kernel_cols;
		const float *_top;
		if(kern_len==5)
		{
			for(int map=0; map<maps; map++) // how many maps  maps= node.chans
			{
				const float *_delta =&delta.x[map*map_stride];
				for(int k=0; k<top.node.chans; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_top = &top.node.x[k*kstep];
					const int w_i = (map+k*maps)*kernel_size;
					const float *_t=_top;
					float *_w=dw.x+w_i;
					_w[0]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[1]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[2]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[3]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[4]+= unwrap_2d_dot( _t, _delta,	node_size,top_node_size, delta_size);
					_t=_top+jstep;
					_w=dw.x+w_i+kern_len;
					_w[0]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[1]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[2]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[3]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[4]+= unwrap_2d_dot( _t, _delta,	node_size,top_node_size, delta_size);
					_t=_top+jstep*2;
					_w=dw.x+w_i+kern_len*2;
					_w[0]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[1]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[2]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[3]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[4]+= unwrap_2d_dot( _t, _delta,	node_size,top_node_size, delta_size);
					_t=_top+jstep*3;
					_w=dw.x+w_i+kern_len*3;
					_w[0]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[1]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[2]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[3]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[4]+= unwrap_2d_dot( _t, _delta,	node_size,top_node_size, delta_size);
					_t=_top+jstep*4;
					_w=dw.x+w_i+kern_len*4;
					_w[0]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[1]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[2]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[3]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[4]+= unwrap_2d_dot( _t, _delta,	node_size,top_node_size, delta_size);
				} //y
			} // all maps=chans 
		}
		else if(kern_len==3)
		{
			for(int map=0; map<maps; map++) // how many maps  maps= node.chans
			{
				const float *_delta =&delta.x[map*map_stride];
				for(int k=0; k<top.node.chans; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_top = &top.node.x[k*kstep];
					const int w_i = (map+k*maps)*kernel_size;
					dw.x[w_i+0+(0)*kern_len]+= unwrap_2d_dot( _top + 0+(0)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+1+(0)*kern_len]+= unwrap_2d_dot( _top + 1+(0)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+2+(0)*kern_len]+= unwrap_2d_dot( _top + 2+(0)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+0+(1)*kern_len]+= unwrap_2d_dot( _top + 0+(1)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+1+(1)*kern_len]+= unwrap_2d_dot( _top + 1+(1)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+2+(1)*kern_len]+= unwrap_2d_dot( _top + 2+(1)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+0+(2)*kern_len]+= unwrap_2d_dot( _top + 0+(2)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+1+(2)*kern_len]+= unwrap_2d_dot( _top + 1+(2)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+2+(2)*kern_len]+= unwrap_2d_dot( _top + 2+(2)*jstep, _delta,	node_size,top_node_size, delta_size);
				} //y
			} // all maps=chans 
		}
		else
		{
		
			for(int map=0; map<maps; map++) // how many maps  maps= node.chans
			{
				const float *_delta =&delta.x[map*map_stride];
				for(int k=0; k<top.node.chans; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_top = &top.node.x[k*kstep];
					const int w_i = (map+k*maps)*kernel_size;
					for(int jj=0; jj<kern_len; jj+=1)
					{
						for(int ii=0; ii<kern_len; ii+=1)
						{
							dw.x[w_i+ii+(jj)*kern_len]+= unwrap_2d_dot( _top + ii+(jj)*jstep, _delta,
								node_size,top_node_size, delta_size);

						} // all input chans
					} // x
				} //y
			} // all maps=chans 
		}
	}

#endif
};

//----------------------------------------------------------------------------------------------------------
// D E E P C N E T   
// 2x2 convolution followed by 2x2 max pool
// odd number should be in-size, then -1 after convolution and divide by 2 for output size
class deepcnet_layer : public base_layer
{
	int _stride;
	matrix conv_delta;
	std::vector<int> _max_map;
public:
	int kernel_rows;
	int kernel_cols;
	int maps;
	//int maps_per_kernel;
	int kernels_per_map;
	static const int _pool=2;


	deepcnet_layer(const char *layer_name, int _c, activation_function *p) : base_layer(layer_name, 2, 2, _c)
	{
		p_act = p; _stride = 1; kernel_rows = 2; kernel_cols = 2; maps = _c; 
		kernels_per_map = 0; pad_cols = 1; pad_rows = 1;
		_use_bias = true;
	}
	virtual  ~deepcnet_layer() {}
	virtual std::string get_config_string() { std::string str = "deepcnet " + int2str(maps) + " " + p_act->name + "\n"; return str; }

	virtual int fan_size() { return kernel_rows*kernel_cols*maps *kernels_per_map; }

	virtual void resize(int _w, int _h = 1, int _c = 1) // special resize nodes because bias handled differently with shared wts
	{
		node = matrix(_w, _h, _c);
		bias = matrix(1, 1, _c);
		bias.fill(0.);
		_max_map.resize(_w*_h*_c);

		conv_delta = matrix(_w*_pool, _h*_pool, maps);

#ifndef MOJO_NO_TRAINING
		delta = matrix(_w, _h, _c, NULL, true);
#endif
	}

	// this connection work won't work with multiple top layers (yet)
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
		top.forward_linked_layers.push_back(std::make_pair(weight_mat_index, this));
#ifndef MOJO_NO_TRAINING
		backward_linked_layers.push_back(std::make_pair(weight_mat_index, &top));
#endif
		// re-shuffle these things so weights of size kernel w,h,kerns - node of size see below
		//int total_kernels=top.node.chans*node.chans;
		kernels_per_map += top.node.chans;
		resize((top.node.cols - 1) / _pool, (top.node.rows - 1) / _pool, maps);
		return new matrix(kernel_cols, kernel_rows, maps*kernels_per_map);
	}

	// activate_nodes
	virtual void activate_nodes()
	{ 
		const int map_size = node.rows*node.cols;
		const int map_stride = node.chan_stride;
		const int _maps = maps;

		MOJO_THREAD_THIS_LOOP(_thread_count)
		for (int c=0; c<_maps; c++) p_act->fc(&node.x[c*map_stride],map_size,bias.x[c]);
	}

	virtual void accumulate_signal(const base_layer &top, const matrix &w, const int train = 0)
	{
		const int kstep = top.node.chan_stride;
		const int jstep = top.node.cols;
		//int output_index=0;
		const int kernel_size = kernel_cols*kernel_rows;
		const int kernel_map_step = kernel_size*kernels_per_map;
		
		const int pool_map_size = node.cols*node.rows;
		const int pool_map_stride = node.chan_stride;
		const float *_w = w.x;
		const int top_chans = top.node.chans;
		const int map_cnt = maps;
		const int w_size = kernel_cols;
		const int stride = _stride; 

		const int conv_size = node.cols * _pool;
		const int pool_size = node.cols;
		const int top_node_size = top.node.cols;
		const int outsize = pool_size*pool_size;
		int *p_map = _max_map.data();

		matrix imgsum_ptr(jstep-1,jstep-1,maps,NULL,true);
		imgsum_ptr.fill(0);

		matrix img_ptr( top.node.cols,  top.node.cols, 2*2, NULL, true);
//#pragma omp parallel for schedule(guided) num_threads(_thread_count)
		for (int k = 0; k < top_chans; k++) // input channels --- same as kernels_per_map - kern for each input
		{
			unwrap_aligned_NxN(2, img_ptr.x, &top.node.x[k*kstep], jstep, 1);

//			MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
MOJO_THREAD_THIS_LOOP(_thread_count)
			for (int map = 0; map < map_cnt; map+=1) // how many maps  maps= node.chans
			{
				//std::cout << omp_get_thread_num();
				float *out = imgsum_ptr.x  + imgsum_ptr.chan_stride*map;
				dotsum_unwrapped_2x2(img_ptr.x, &w.x[(map + k*maps)*kernel_size], out, (jstep-1)*(jstep-1));
			}
		}
		int idx = 0;
		for (int map = 0; map < map_cnt; map++) // how many maps  maps= node.chans
		{
			float *out = node.x + pool_map_stride*map;
			float *sum = imgsum_ptr.x + imgsum_ptr.chan_stride*map;
			int cnt=0;
			for (int j = 0; j < conv_size; j += _pool)
			{
				for (int i = 0; i < conv_size; i += _pool)
				{
					int maxi = i + j*conv_size;
					if (sum[maxi] < sum[i + 1 + j*conv_size])
						maxi = i + 1 + j*conv_size;
					if (sum[maxi] < sum[i + (j + 1)*conv_size])
						maxi = i + (j + 1)*conv_size;
					if (sum[maxi] < sum[i + 1 + (j + 1)*conv_size])
						maxi = i + 1 + (j + 1)*conv_size;
						
					//const int pool_idx = (i + j * pool_size) / _pool;
					out[cnt] = sum[maxi];
					p_map[idx] = maxi+ conv_size*conv_size*map;
					idx++;
					cnt++;
				}
			}
		}
	}


#ifndef MOJO_NO_TRAINING

	// convolution::distribute_delta
	virtual void distribute_delta(base_layer &top, const matrix &w, const int train = 1)
	{

		// here to calculate top_delta += bottom_delta * W
		//		top_delta.x[s] += bottom_delta.x[t]*w.x[s+t*w.cols];

		const int kstep = top.delta.chan_stride;
		const int jstep = top.delta.cols;
		const int output_index = 0;
		const int kernel_size = kernel_cols*kernel_rows;
		const int kernel_map_step = kernel_size*kernels_per_map;
		const float *_w = w.x;
		const int w_size = kernel_cols;
		const int map_cnt = maps;
		const int top_delta_size = top.delta.rows;
		const int top_delta_chans = top.delta.chans;
		const int stride = _stride;

		//mojo::matrix intermediate_delta(delta.cols * 2, delta.rows * 2, delta.chans);
		conv_delta.fill(0);

		int *p_map = _max_map.data();
		const int s = (int)_max_map.size();

		// put the maxpool result
		for (int k = 0; k<s; k++) conv_delta.x[p_map[k]] += delta.x[k];

//		std::cout << "deepc max";
//		for (int i = 0; i < 10; i++) std::cout << delta.x[i] << ",";
///		std::cout << "topc max";
//		for (int i = 0; i < 10; i++) std::cout << conv_delta.x[i] << ",";


		matrix delta_pad(conv_delta, pad_cols, pad_rows);

		const int map_size = delta_pad.cols*delta_pad.rows;
		const int map_stride = delta_pad.chan_stride;
		const int delta_size = delta_pad.cols;



			matrix img_ptr(delta_size, delta_size, 4, NULL, true);
			matrix filter_ptr(4, 1);
		matrix delt(top.delta.cols, top.delta.rows, top.delta.chans,NULL,true);
					
			//matrix imgout_ptr(outsize + 7, 1);
			for (int map = 0; map < map_cnt; map+=1) // how many maps  maps= node.chans
			{
				unwrap_aligned_NxN(2, img_ptr.x, &delta_pad.x[map*map_stride], delta_size, stride);
				const int outsize = top_delta_size*top_delta_size;
				for (int k = 0; k < top_delta_chans; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_w = &w.x[(k*maps + map)*kernel_size];
					// flip-flip to make 180 version
					for (int ii = 0; ii < 4; ii++) filter_ptr.x[ii] = _w[3 - ii];
					//float *out = node.x + map_stride*map;
							float *out = &delt.x[k*delt.chan_stride];
							memcpy(out,&top.delta.x[k*kstep],sizeof(float)*outsize);
							dotsum_unwrapped_2x2(img_ptr.x, filter_ptr.x, out, outsize);// imgout_ptr.x, outsize);
							memcpy(&top.delta.x[k*kstep],out,sizeof(float)*outsize);

//					float *out = &top.delta.x[k*kstep];
	//				dotsum_unwrapped_2x2(img_ptr.x, filter_ptr.x, out, outsize);// imgout_ptr.x, outsize);
				}
			}



	}


	// convolution::calculate_dw
	virtual void calculate_dw(const base_layer &top, matrix &dw, const int train = 1)
	{
		int kstep = top.delta.cols*top.delta.rows;
		int jstep = top.delta.cols;
		int output_index = 0;
		int kernel_size = kernel_cols*kernel_rows;
		int kernel_map_step = kernel_size*kernels_per_map;
		int map_size = conv_delta.cols*conv_delta.rows;

		dw.resize(kernel_cols, kernel_rows, kernels_per_map*maps);
		dw.fill(0);

		// node x already init to 0
		output_index = 0;
		const int stride = _stride;
		const int top_node_size = top.node.cols;
		const int delta_size = conv_delta.cols;
		const int kern_len = kernel_cols;
		const float *_top;


		for (int map = 0; map<maps; map++) // how many maps  maps= node.chans
		{
			const float *_delta = &conv_delta.x[map*map_size];
			for (int k = 0; k<top.node.chans; k++) // input channels --- same as kernels_per_map - kern for each input
			{
				_top = &top.node.x[k*kstep];
				const int w_i = (map + k*maps)*kernel_size;
				for (int jj = 0; jj<kern_len; jj += 1)
				{
					for (int ii = 0; ii<kern_len; ii += 1)
					{
						dw.x[w_i + ii + (jj)*kern_len] +=
							unwrap_2d_dot(_top + ii + (jj)*jstep, _delta, 
								delta_size, top_node_size, delta_size);

					} // all input chans
				} // x
			} //y
		} // all maps=chans

	}

#endif
};

//----------------------------------------------------------------------------------------------------------
// C O N C A T E N A T I O N   |     R E S I Z E    |      P  A  D 
//
// puts a set of output maps together and pads to the desired size
class concatenation_layer : public base_layer
{
	std::map<const base_layer*, int> layer_to_channel;  // name-to-index of layer for layer management

	int _maps;
	mojo::pad_type _pad_type;
public:
	concatenation_layer(const char *layer_name, int _w, int _h, mojo::pad_type p= mojo::zero) : base_layer(layer_name, _w, _h)
	{
		_maps = 0;
		_pad_type = p;
		_has_weights = false;
		p_act = NULL;// new_activation_function("identity");
	}
	virtual  ~concatenation_layer() {}
	virtual std::string get_config_string() 
	{ 
		std::string str_p = " zero\n";
		if (_pad_type == mojo::edge) str_p = " edge\n";
		else if (_pad_type == mojo::median_edge) str_p = " median_edge\n";

		std::string str = "concatenate " + int2str(node.cols) + str_p;
		return str; 
	}
	// this connection work won't work with multiple top layers (yet)
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
		//if (layer_to_channel[&top]) bail("layer already addded to pad layer"); //already exists
		layer_to_channel[&top] = _maps;
		_maps += top.node.chans;
		resize(node.cols, node.rows, _maps);
		return base_layer::new_connection(top, weight_mat_index);
	}

	// no weights 
	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train = 1) {}

	virtual void accumulate_signal(const base_layer &top, const matrix &w, const int train = 0)
	{
		const float *top_node = top.node.x;
		const int size = node.rows*node.cols;
	
		int opadx = node.cols - top.node.cols;
		int opady = node.rows - top.node.rows;
		int padx=0, pady=0, padx_ex=0, pady_ex=0;

		if (opadx > 0) padx = opadx/2;
		if (opady > 0) pady = opady/2;

		if (opadx % 2 != 0) {
			padx_ex = 1;
		}
		if (opady % 2 != 0) {
			pady_ex = 1;
		}

		int map_offset = layer_to_channel[&top];

		if (padx+ padx_ex > 0 || pady+ pady_ex > 0 )
		{
			matrix m = top.node.pad(padx, pady, padx+ padx_ex, pady+pady_ex, _pad_type, _thread_count);
			memcpy(node.x + node.chan_stride*map_offset, m.x, sizeof(float)*m.size());
		}
		else if((node.cols == top.node.cols) && (node.rows == top.node.rows))
		{
			memcpy(node.x + node.chan_stride*map_offset, top.node.x, sizeof(float)*top.node.size());
		}
		else
		{
			// crop
			int dx = abs(padx) / 2;
			int dy = abs(pady) / 2;
			matrix m = top.node.crop(dx, dy, node.cols, node.rows, _thread_count);
			memcpy(node.x + node.chan_stride*map_offset, m.x, sizeof(float)*m.size());
		}
	}
#ifndef MOJO_NO_TRAINING

	virtual void distribute_delta(base_layer &top, const matrix &w, const int train = 1)
	{
		int map_offset = layer_to_channel[&top];
		int padx = node.cols - top.node.cols;
		int pady = node.rows - top.node.rows;
		if (padx > 0) padx /= 2;
		if (pady > 0) pady /= 2;

		if (padx > 0 || pady > 0)
		{
			matrix m = delta.get_chans(map_offset, top.delta.chans);
			top.delta += m.crop(padx, pady, top.delta.cols, top.delta.rows);
		}
		else if ((node.cols == top.node.cols) && (node.rows == top.node.rows))
		{
			top.delta += delta.get_chans(map_offset, top.delta.chans);
		}
		else
		{
			matrix m = delta.get_chans(map_offset, top.delta.chans);
			// pad
			int dx = abs(padx) / 2;
			int dy = abs(pady) / 2;
			top.delta += m.pad(dx, dy);

		}

	}
#endif
};

//--------------------------------------------------
// N E W    L A Y E R 
//
// "input", "fully_connected","max_pool","convolution","concatination"
base_layer *new_layer(const char *layer_name, const char *config)
{
	std::istringstream iss(config); 
	std::string str;
	iss>>str;
	int w,h,c,s;
	if(str.compare("input")==0)
	{
		iss>>w; iss>>h; iss>>c;
		return new input_layer(layer_name, w,h,c);
	}
	else if(str.compare("fully_connected")==0)
	{
		std::string act;
		iss>>c; iss>>act; 
		return new fully_connected_layer(layer_name, c, new_activation_function(act));
	}
	else if (str.compare("softmax") == 0)
	{
		//std::string act;
		iss >> c; //iss >> act;
		return new fully_connected_layer(layer_name, c, new_activation_function("softmax"));
	}
	else if(str.compare("max_pool")==0)
	{
		iss >> c;  iss >> s;
		if(s>0 && s<=c)
			return new max_pooling_layer(layer_name, c, s);
		else
			return new max_pooling_layer(layer_name, c);
	}
	else if (str.compare("mfm") == 0)
	{
		iss >> c;
		return new maxout_layer(layer_name, c);
	}
	/*
	else if (str.compare("activation") == 0)
	{
		iss >> s;
		return new activation_layer(layer_name, s);
	}
	*/
	else if (str.compare("semi_stochastic_pool") == 0)
	{
		iss >> c;  iss >> s;
		if (s>0 && s <= c)
			return new semi_stochastic_pooling_layer(layer_name, c, s);
		else
			return new semi_stochastic_pooling_layer(layer_name, c);
	}

	else if (str.compare("deepcnet") == 0)
	{
		std::string act;
		iss >> c; iss >> act;
		return new deepcnet_layer(layer_name, c, new_activation_function(act));
	}
	else if(str.compare("convolution")==0)
	{
		std::string act;
		iss>>w;iss>>c; iss >> s; iss>>act;
		return new convolution_layer(layer_name, w,c,s, new_activation_function(act));
	}
	else if (str.compare("dropout") == 0)
	{
		float fc;
		iss >> fc;
		return new dropout_layer(layer_name, fc);
	}
	else if((str.compare("resize")==0) || (str.compare("concatenate") == 0))
	{
		std::string pad;
		iss>>w; 
		iss >> pad;
		mojo::pad_type p = mojo::zero;
		if (pad.compare("median") == 0) p = mojo::median_edge;
		else if (pad.compare("median_edge") == 0) p = mojo::median_edge;
		else if (pad.compare("edge") == 0) p = mojo::edge;

		return new concatenation_layer(layer_name, w,w, p);
	}	
	else
	{

		//fprintf(stderr, "ERROR : layer type not valid: '%s'", str);
		bail("ERROR : layer type not valid: '" + str + "'\n");
	}

	return NULL;
}


} // namespace
