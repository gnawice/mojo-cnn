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

#define int2str(a) std::to_string((long long)a)
#define float2str(a) std::to_string((long double)a)

//----------------------------------------------------------------------------------------------------------
// B A S E   L A Y E R
//
// all other layers derived from this
class base_layer 
{
public:
	activation_function *p_act;
	
	int pad_cols, pad_rows;
	matrix node;
	matrix bias; // this is something that maybe should be in the same class as the weights... but whatever. handled differently for different layers
	
	std::string name;
	// index of W matrix, index of connected layer
	std::vector<std::pair<int,base_layer*>> forward_linked_layers;
#ifndef NO_TRAINING_CODE
	matrix delta;
	std::vector<std::pair<int,base_layer*>> backward_linked_layers;

	virtual void distribute_delta(base_layer &top, const matrix &w, const int train = 1) =0;
	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train =1)=0;
#endif
	virtual void accumulate_signal(const base_layer &top_node, const matrix &w, const int train =0) =0;

	base_layer(const char* layer_name, int _w, int _h=1, int _c=1) : node(_w, _h, _c), bias(_w, _h, _c), p_act(NULL), name(layer_name), pad_cols(0), pad_rows(0)
		#ifndef NO_TRAINING_CODE
		,delta(_w,_h,_c)
		#endif
	{bias.fill(0.);}


	virtual void resize(int _w, int _h=1, int _c=1)
	{
		node =matrix(_w,_h,_c);
		bias =matrix(_w,_h,_c);
		bias.fill(0.);
		#ifndef NO_TRAINING_CODE
		delta =matrix(_w,_h,_c);
		#endif
	}
	
	virtual ~base_layer(){if(p_act) delete p_act;}
	virtual int fan_size() {return node.chans*node.rows*node.cols;}

	virtual void activate_nodes() {for (int i=0; i<node.size(); i++)  node.x[i]=p_act->f(node.x, i, node.size(), bias.x[i]);}
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
		top.forward_linked_layers.push_back(std::make_pair((int)weight_mat_index,this));
		#ifndef NO_TRAINING_CODE
		backward_linked_layers.push_back(std::make_pair((int)weight_mat_index,&top));
		#endif
		int rows=node.cols*node.rows*node.chans; 
		int cols=top.node.cols*top.node.rows*top.node.chans; 
		return new matrix(cols,rows, 1);
	}

	inline float f(float *in, int i, int size, float bias) {return p_act->f(in, i, size, bias);};
	inline float df(float *in, int i, int size) {return p_act->df(in, i, size);};
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
	virtual void activate_nodes() {}
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
	fully_connected_layer(const char *layer_name, int _size, activation_function *p ) : base_layer(layer_name,_size,1,1)  {p_act=p; }//layer_type=fully_connected_type;}
	virtual std::string get_config_string() {std::string str="fully_connected "+int2str(node.size())+ " "+p_act->name+"\n"; return str;}
	virtual void accumulate_signal( const base_layer &top,const matrix &w, const int train =0)
	{
		// doesn't care if shape is not 1D
		// here weights are formated in matrix, top node in cols, bottom node along rows. (note that my top is opposite of traditional understanding)
		node += top.node.dot_1dx2d(w);
//		const int s = w.rows;
//		const int ts = top.node.size();
//		for (int j = 0; j<s; j++)	
//			node.x[j] += dot(top.node.x, w.x+j*w.cols, ts);

	}
#ifndef NO_TRAINING_CODE
	virtual void distribute_delta(base_layer &top, const matrix &w, const int train =1)
	{
		const int w_cols = w.cols;
		for (int b = 0; b < delta.size(); b++)
		{
			const float cb = delta.x[b];
			for (int t = 0; t < top.delta.size(); t++) top.delta.x[t] += cb*w.x[t + b*w_cols];
		}
	}

	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train = 1)
	{
		const float *bottom = delta.x; const int sizeb = delta.size();
		const float *top = top_layer.node.x; const int sizet = top_layer.node.size();
		dw.resize(sizet, sizeb, 1);

		for (int b = 0; b < sizeb; b++)
		{
			const float cb = bottom[b];
			for (int t = 0; t < sizet; t++)	dw.x[t + b*sizet] = top[t] * cb;
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
	max_pooling_layer(const char *layer_name, int pool_size, activation_function *p = NULL) : base_layer(layer_name, 1)
	{
		p_act = p; _stride = pool_size; _pool_size = pool_size; p_act = new_activation_function("identity"); //layer_type=pool_type;
	}
	max_pooling_layer(const char *layer_name, int pool_size, int stride, activation_function *p=NULL ) : base_layer(layer_name, 1)
	{
		p_act=p; _stride= stride; _pool_size=pool_size; p_act=new_activation_function("identity"); //layer_type=pool_type;
	}
	virtual  ~max_pooling_layer(){}
	virtual std::string get_config_string() {std::string str="max_pool "+int2str(_pool_size) +" "+ int2str(_stride) +"\n"; return str;}

	// ToDo would like delayed activation of conv layer if available
	virtual void activate_nodes(){ return;}
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
		// wasteful to add weight matrix (1x1x1), but makes other parts of code more OO
		// bad will happen if try to put more than one pool layer
		top.forward_linked_layers.push_back(std::make_pair(weight_mat_index,this));
		int pool_size= _pool_size;
		int w = (top.node.cols) / pool_size;
		int h = (top.node.rows) / pool_size;
		if (_stride != _pool_size)
		{
			w = 1+((top.node.cols - _pool_size) / _stride);
			h = 1+((top.node.rows - _pool_size) / _stride);
		}
					
//			resize((top.node.cols-2*pad_cols)/pool_size, (top.node.rows-2*pad_rows)/pool_size, top.node.chans);
		resize(w,h, top.node.chans);
#ifndef NO_TRAINING_CODE
		backward_linked_layers.push_back(std::make_pair(weight_mat_index,&top));
#endif
		return new matrix(1,1,1);
	}

	// this is downsampling
	virtual void accumulate_signal(const base_layer &top,const matrix &w,const int train =0)
	{
		int kstep=top.node.cols*top.node.rows;
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
						node.x[output_index] = top_node[max_i];
						p_map[output_index] = max_i;
					output_index++;
				}
			}
		}
	}
#ifndef NO_TRAINING_CODE

	// this is upsampling
	virtual void distribute_delta(base_layer &top, const matrix &w, const int train =1)
	{
		int *p_map = _max_map.data();
		for(int k=0; k<(int)_max_map.size(); k++) 
			top.delta.x[p_map[k]]+=delta.x[k];
	}
#endif
};

//----------------------------------------------------------------------------------------------------------
// S E M I   S T O C H A S T I C   P O O L I N G   
// concept similar to stochastic pooling but only slects 'max' based on top 2 candidates
class semi_stochastic_pooling_layer :  public max_pooling_layer
{
public:
	semi_stochastic_pooling_layer(const char *layer_name, int pool_size, activation_function *p = NULL) : max_pooling_layer(layer_name, pool_size)
	{
	//	p_act = p; _stride = pool_size; _pool_size = pool_size; p_act = new_activation_function("identity"); //layer_type=pool_type;
	}
	semi_stochastic_pooling_layer(const char *layer_name, int pool_size, int stride, activation_function *p = NULL) : max_pooling_layer(layer_name, pool_size, stride)
	{
		//p_act = p; _stride = stride; _pool_size = pool_size; p_act = new_activation_function("identity"); //layer_type=pool_type;
	}
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
// dropout applies to the previous layer, so dropout can be you last layer
class dropout_layer : public base_layer
{
	float _dropout_rate;
	matrix drop_mask;
public:
	dropout_layer(const char *layer_name, float dropout_rate, activation_function *p = NULL) : base_layer(layer_name, 1)
	{
		_dropout_rate = dropout_rate;
		p_act = p; p_act = new_activation_function("identity");
	}
	virtual  ~dropout_layer() {}
	virtual std::string get_config_string() { std::string str = "dropout " + float2str(_dropout_rate)+"\n"; return str; }
	virtual void resize(int _w, int _h = 1, int _c = 1)
	{
		if (_w<1) _w = 1; if (_h<1) _h = 1; if (_c<1) _c = 1;
		drop_mask.resize(_w, _h, _c);
		base_layer::resize(_w, _h, _c);
	}

	virtual void activate_nodes() { return; }
	// no weights 
	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train = 1) {}
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
		// wasteful to add weight matrix (1x1x1), but makes other parts of code more OO
		// bad will happen if try to put more than one pool layer
		top.forward_linked_layers.push_back(std::make_pair(weight_mat_index, this));
		int pool_size = 1;
		int w = (top.node.cols) / 1;
		int h = (top.node.rows) / 1;
		resize(w, h, top.node.chans);
#ifndef NO_TRAINING_CODE
		backward_linked_layers.push_back(std::make_pair(weight_mat_index, &top));
#endif
		return new matrix(1, 1, 1);
	}

	// for dropout...
	// we know this is called first in the backward pass, and the train will be set to 1
	// when that happens the dropouts will be set. 
	// different dropouts for each mininbatch... don't know if that matters...
	virtual void accumulate_signal(const base_layer &top, const matrix &w, const int train = 0)
	{
		const float *top_node = top.node.x;
		const int size = top.node.chans*top.node.rows*top.node.cols;

		if (train)
		{
			for (int k = 0; k < size; k++)
			{
				int r = rand() % 100;
				if (r <= (_dropout_rate*100.f))
					drop_mask.x[k] = 0.5;
				else
					drop_mask.x[k] = 1.0;
				node.x[k] = top_node[k] * drop_mask.x[k];
				//		p_map[output_index] = max_i;
				//output_index++;
			}
		}
		else
		{
			for (int k = 0; k < size; k++)
				node.x[k] = top_node[k];
		}
	}
#ifndef NO_TRAINING_CODE

	virtual void distribute_delta(base_layer &top, const matrix &w, const int train = 1)
	{
		for (int k = 0; k<top.node.chans*top.node.rows*top.node.cols; k++)
		{
			if(drop_mask.x[k]==1)
				top.delta.x[k] += delta.x[k];
		}
	}
#endif
};

//----------------------------------------------------------------------------------------------------------
// M A X   O U T
// 
// dropout applies to the previous layer, so dropout can be you last layer
class maxout_layer : public base_layer
{
	int _pool_group;
	matrix max_map;
public:
	maxout_layer(const char *layer_name, int  pool_group, activation_function *p = NULL) : base_layer(layer_name, 1)
	{
		_pool_group = pool_group;
		p_act = p; p_act = new_activation_function("identity");
	}
	virtual  ~maxout_layer() {}
	virtual std::string get_config_string() { std::string str = "maxout_map_pool" + int2str(_pool_group) + "\n"; return str; }
	virtual void resize(int _w, int _h = 1, int _c = 1)
	{
		_c /= 2;
		if (_w<1) _w = 1; if (_h<1) _h = 1; if (_c<1) _c = 1;
		max_map.resize(_w, _h, _c);
		base_layer::resize(_w, _h, _c);
	}

	virtual void activate_nodes() { return; }
	// no weights 
	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train = 1) {}
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
		// wasteful to add weight matrix (1x1x1), but makes other parts of code more OO
		// bad will happen if try to put more than one pool layer
		top.forward_linked_layers.push_back(std::make_pair(weight_mat_index, this));
		int pool_size = 1;
		int w = (top.node.cols) / 1;
		int h = (top.node.rows) / 1;
		resize(w, h, top.node.chans);
#ifndef NO_TRAINING_CODE
		backward_linked_layers.push_back(std::make_pair(weight_mat_index, &top));
#endif
		return new matrix(1, 1, 1);
	}

	// for maxout
	// we know this is called first in the backward pass, and the train will be set to 1
	// when that happens the dropouts will be set. 
	// different dropouts for each mininbatch... don't know if that matters...
	virtual void accumulate_signal(const base_layer &top, const matrix &w, const int train = 0)
	{
		const float *top_node = top.node.x;
		const int chan_size = top.node.rows*top.node.cols;

		for (int c = 0; c < top.node.chans; c+=2)
		{
			for (int i = 0; i < chan_size; i++)
			{
				if (top.node.x[i + c*chan_size] > top.node.x[i + (c + 1)*chan_size])
				{
					node.x[i + c/2*chan_size] = top.node.x[i + c*chan_size];
					max_map.x[i + c/2*chan_size] = 0;
				}
				else
				{
					node.x[i + c/2*chan_size] = top.node.x[i + (c + 1)*chan_size];
					max_map.x[i + c/2*chan_size] = 1;
				}
			}
		}
	}
#ifndef NO_TRAINING_CODE

	virtual void distribute_delta(base_layer &top, const matrix &w, const int train = 1)
	{
		const int chan_size = node.cols*node.rows;
		for (int c = 0; c < node.chans; c++)
		{
			for (int k = 0; k < node.cols*node.rows; k++)
			{
				int maxmap= max_map.x[k + c*chan_size];
//				top.delta.x[k + chan_size*c] += delta.x[k + c*chan_size];

				top.delta.x[k+ chan_size*(2*c+ maxmap)] += delta.x[k+c*chan_size];
			}
		}
	}
#endif
};

//----------------------------------------------------------------------------------------------------------
// F R A C T I O N A L    M A X   P O O L I N G   
//  - not working yet
// may split to max and ave pool class derived from pooling layer.. but i never use ave pool anymore
class fractional_max_pooling_layer : public max_pooling_layer
{
	float _fpool_size;
	int _out_size;
public:
	int stride;
	fractional_max_pooling_layer(const char *layer_name, int out_size, activation_function *p=NULL ) : max_pooling_layer(layer_name,-1,p)
	{
		stride=-1; _out_size=out_size; _fpool_size =-1.f;
	}
	virtual std::string get_config_string() {std::string str="fractional_max_pool "+int2str(_out_size)+"\n"; return str;}

	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
			// wasteful to add weight matrix (1x1x1), but makes other parts of code more OO
			// bad will happen if try to put more than one pool layer
			top.forward_linked_layers.push_back(std::make_pair(weight_mat_index,this));
			//int pool_size=stride;
			_fpool_size = (float)top.node.cols/(float)_out_size;
			stride = (int)_fpool_size; // trunc - so will be less than needed
			resize(_out_size,_out_size, top.node.chans);

#ifndef NO_TRAINING_CODE
			backward_linked_layers.push_back(std::make_pair(weight_mat_index,&top));
#endif
			return new matrix(1,1,1);
	}

	// this is downsampling
	virtual void accumulate_signal(const base_layer &top,const matrix &w,const int train =0)
	{
		const int top_node_size=top.node.cols*top.node.rows;
		const int top_node_len=top.node.cols;
		int output_index=0;
		int *p_map = _max_map.data();
		
		const float *top_node = top.node.x;

		int pool_y=(int)_fpool_size; if(top.node.rows==1) pool_y=1; //-top.pad_rows*2==1) pool_y=1;
		int pool_x=(int)_fpool_size; if(top.node.cols==1) pool_x=1;//-top.pad_cols*2==1) pool_x=1;
		int _int_pool=(int)_fpool_size;  // will truncate to int
		//int delta_stride_i=rand()%2;
		//int delta_stride_j=rand()%2;
		// should probably throw() if _pool_size<1 
		float left_over= _fpool_size -(float)_int_pool;
		for(int k=0; k<node.chans; k++)
		{
			for(int j=0; j<node.rows; j++)
			{
				int rand_left_over_j=0;//((int)(j*left_over))%2;//(rand()%100)/300.f;
				int jj=(int)(_fpool_size*j);//+(int)((float)_int_pool/2.*rand_left_over_j);
				for(int i=0; i<node.cols; i++)
				{
					float rand_left_over_i=0;//((int)((i+j)*left_over))%2;//left_over*(rand()%100)/300.f;
					int ii=(int)(_fpool_size*i);//+(int)((float)_int_pool/2.*rand_left_over_i);
				
					int max_i=0;
					float max=0;
			
					if(_fpool_size <=2.f)  // pools will be 2 with step 1 or 2
					{
						if(ii>=top.node.cols-2) ii=top.node.cols-2;
						if(jj>=top.node.cols-2) jj=top.node.rows-2;
					const int base_index=ii+jj*top_node_len+k*top_node_size;
					max_i=base_index;
					max=top_node[base_index];
						const float *n=top_node+base_index;
						//if(max<n[0]) { max = n[0]; max_i=max_i;}
						if(max<n[1]) { max = n[1]; max_i=base_index+1;}
						n+=top_node_len;
						if(max<n[0]) { max = n[0]; max_i=base_index+top_node_len;}
						if(max<n[1]) { max = n[1]; max_i=base_index+top_node_len+1;}
					}
					else if(_fpool_size <=3.f) // pools will be 3 with step 2 or 3
					{
						if(ii>=top.node.cols-3) ii=top.node.cols-3;
						if(jj>=top.node.cols-3) jj=top.node.rows-3;
					const int base_index=ii+jj*top_node_len+k*top_node_size;
					max_i=base_index;
					max=top_node[base_index];
						const float *n=top_node+base_index;
						//if(max<n[0]) { max = n[0]; max_i=max_i;}
						if(max<n[1]) { max = n[1]; max_i=base_index+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+2;}
						n+=top_node_len;
						if(max<n[0]) { max = n[0]; max_i=base_index+top_node_len;}
						if(max<n[1]) { max = n[1]; max_i=base_index+top_node_len+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+top_node_len+2;}
						n+=top_node_len;
						if(max<n[0]) { max = n[0]; max_i=base_index+2*top_node_len;}
						if(max<n[1]) { max = n[1]; max_i=base_index+2*top_node_len+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+2*top_node_len+2;}
					}
					else if(_fpool_size <=4.f) // pools will be 4 with step 3 or 4
					{
						if(ii>=top.node.cols-4) ii=top.node.cols-4;
						if(jj>=top.node.cols-4) jj=top.node.rows-4;
					const int base_index=ii+jj*top_node_len+k*top_node_size;
					max_i=base_index;
					max=top_node[base_index];
						const float *n=top_node+base_index;
						//if(max<n[0]) { max = n[0]; max_i=max_i;}
						if(max<n[1]) { max = n[1]; max_i=base_index+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+2;}
						if(max<n[3]) { max = n[3]; max_i=base_index+3;}
						n+=top_node_len;
						if(max<n[0]) { max = n[0]; max_i=base_index+top_node_len;}
						if(max<n[1]) { max = n[1]; max_i=base_index+top_node_len+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+top_node_len+2;}
						if(max<n[3]) { max = n[3]; max_i=base_index+top_node_len+3;}
						n+=top_node_len;
						if(max<n[0]) { max = n[0]; max_i=base_index+2*top_node_len;}
						if(max<n[1]) { max = n[1]; max_i=base_index+2*top_node_len+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+2*top_node_len+2;}
						if(max<n[3]) { max = n[3]; max_i=base_index+2*top_node_len+3;}
						n+=top_node_len;
						if(max<n[0]) { max = n[0]; max_i=base_index+3*top_node_len;}
						if(max<n[1]) { max = n[1]; max_i=base_index+3*top_node_len+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+3*top_node_len+2;}
						if(max<n[3]) { max = n[3]; max_i=base_index+3*top_node_len+3;}
					}
					else
					{
						if(ii>=top.node.cols-pool_x) ii=top.node.cols-pool_x;
						if(jj>=top.node.rows-pool_y) jj=top.node.rows-pool_y;
						const int base_index=ii+jj*top_node_len+k*top_node_size;
						max_i=base_index;
						max=top_node[base_index];
						for(int jp=0; jp<pool_y; jp+=1)
						{
							for(int ip=0; ip<pool_x; ip+=1)
							{
								int index=ii+ip+(jj+jp)*top_node_len+k*top_node_size;
								if(max<top_node[index])
								{
									max = top_node[index];
									max_i=index;
								}
							}
						}
					}
					node.x[output_index]=max;
					p_map[output_index]=max_i;
					output_index++;	
				}
			}
		}
	}

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


//	void *filter_mem;
//	void *img_mem;
//	void *imgout_mem;
//	void *img_mem2;
//	void *imgout_mem2;


	convolution_layer(const char *layer_name, int _w, int _h, int _c, activation_function *p ) : base_layer(layer_name, _w, _h, _c) 
	{
		p_act=p; _stride =1; kernel_rows=_h; kernel_cols=_w; maps=_c;kernels_per_map=0; pad_cols = kernel_cols-1; pad_rows = kernel_rows-1;
//		filter_mem = NULL;
//		img_mem = NULL;
//		imgout_mem = NULL;
//		img_mem2 = NULL;
//		imgout_mem2 = NULL;
	}
	virtual  ~convolution_layer() {
//		if (filter_mem) free(filter_mem);
//		if (img_mem) free(img_mem);
//		if (imgout_mem) free(imgout_mem);
//		if (img_mem2) free(img_mem2);
//		if (imgout_mem2) free(imgout_mem2);
	}
	virtual std::string get_config_string() {std::string str="convolution "+int2str(kernel_cols)+" "+int2str(kernel_rows)+" "+int2str(maps)+" "+p_act->name+"\n"; return str;}
	
	virtual int fan_size() { return kernel_rows*kernel_cols*maps *kernels_per_map; }

	
	virtual void resize(int _w, int _h=1, int _c=1) // special resize nodes because bias handled differently with shared wts
	{
		node =matrix(_w,_h,_c); 
		bias =matrix(1,1,_c);
		bias.fill(0.);
		#ifndef NO_TRAINING_CODE
		delta =matrix(_w,_h,_c);
		#endif
	}

	// this connection work won't work with multiple top layers (yet)
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
		top.forward_linked_layers.push_back(std::make_pair(weight_mat_index,this));
		#ifndef NO_TRAINING_CODE
		backward_linked_layers.push_back(std::make_pair(weight_mat_index,&top));
		#endif
		// re-shuffle these things so weights of size kernel w,h,kerns - node of size see below
		//int total_kernels=top.node.chans*node.chans;
		kernels_per_map += top.node.chans;
		resize(top.node.cols-kernel_cols+1, top.node.rows-kernel_rows+1, maps);

		return new matrix(kernel_cols,kernel_rows, maps*kernels_per_map);
	}

	// activate_nodes
	virtual void activate_nodes()
	{ 
		//int total_maps=kernels;
		const int map_size = node.rows*node.cols;
		const int _maps = maps;
		for (int c=0; c<_maps; c++) 
		{
			const float b = bias.x[c];
			float *x= &node.x[c*map_size];

			for (int i=0; i<map_size; i++) x[i]=p_act->f(x,i,map_size,b);
		}
	}


	virtual void accumulate_signal( const base_layer &top, const matrix &w, const int train =0)
	{	
		const int kstep=top.node.cols*top.node.rows;
		const int jstep=top.node.cols;
		//int output_index=0;
		const int kernel_size=kernel_cols*kernel_rows;
		const int kernel_map_step = kernel_size*kernels_per_map;
		const int map_size=node.cols*node.rows;
		const float *_w = w.x;
		const int top_chans = top.node.chans;
		const int map_cnt=maps;
		const int w_size = kernel_cols;
		const int stride = _stride;		
		const int node_size= node.cols;
		const int top_node_size = top.node.cols;
		const int outsize = node_size*node_size;
		
		if(kernel_rows==5)
		{
			// orig implementation
#ifndef MOJO_SSE3
			for (int k = 0; k<top_chans; k++) // input channels --- same as kernels_per_map - kern for each input
			{
				const float *_top_node;
				_top_node = &top.node.x[k*kstep];
				for (int map = 0; map<map_cnt; map++) // how many maps  maps= node.chans
				{
					_w = &w.x[(map + k*maps)*kernel_size];
					for (int j = 0; j < node_size; j += stride)//stride) // input h 
					{
						for (int i = 0; i < node_size; i += stride)//stride) // intput w
						{
							const float v = unwrap_2d_dot_5x5(_top_node + i + j*jstep, _w, jstep, w_size);
							node.x[i + (j)*node_size + map_size*map] += v;
						}
					}
				}
			}
			return; 
#else // MOJO_SSE3
			// ensures 16byte alignment, but maybe not needed if x64
				float* filter_mem = new float [28 + 4];
				float *filter_ptr = (float *)(((uintptr_t)filter_mem + 15) & ~(uintptr_t)0x0F);
				float* img_mem = new float [28 * node_size*node_size+4];
				float *img_ptr = (float *)(((uintptr_t)img_mem + 15) & ~(uintptr_t)0x0F);
				float *imgout_mem = new float [node_size*node_size + 4];
				float *imgout_ptr = (float *)(((uintptr_t)imgout_mem + 15) & ~(uintptr_t)0x0F);
			//		memset(img_ptr, 0, 28*node_size*node_size * sizeof(float));
		//				memset(imgout_ptr, 0, node_size*node_size * sizeof(float));
		//				memset(filter_ptr, 0, 28 * sizeof(float));
				for (int k = 0; k < top_chans; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					unwrap_aligned_5x5(img_ptr, &top.node.x[k*kstep], jstep);

					for (int map = 0; map < map_cnt; map++) // how many maps  maps= node.chans
					{
						memcpy(filter_ptr, &w.x[(map + k*maps)*kernel_size], 25 * sizeof(float));
						dot_unwrapped_5x5_sse(img_ptr, filter_ptr, imgout_ptr, outsize);
						float *out = node.x + map_size*map;
						for (int j = 0; j < outsize; j++) out[j] += imgout_ptr[j];
					}
				}
				delete [] filter_mem;
				delete [] imgout_mem;
				delete [] img_mem;
			return;
			
#endif // MOJO_SSE3
		}
		else if(kernel_rows==3)
		{
#ifndef MOJO_SSE3
			for(int k=0; k<top_chans; k++) // input channels --- same as kernels_per_map - kern for each input
			{
				const float *_top_node;
				_top_node= &top.node.x[k*kstep];
				for(int map=0; map<map_cnt; map++) // how many maps  maps= node.chans
				{
					_w=&w.x[(map+k*maps)*kernel_size];
					for(int j=0; j<node_size; j+= stride)//stride) // input h 
						for(int i=0; i<node_size; i+= stride)//stride) // intput w
						{
							//float v=0;
							const float v=unwrap_2d_dot_3x3(_top_node+i+j*jstep,_w,	jstep,w_size);
							node.x[i+(j)*node_size +map_size*map]+= v;
						}
				}
			}
#else // MOJO_SSE3
			void * filter_mem = malloc(12 * sizeof(float) + 16);
			float *filter_ptr = (float *)(((uintptr_t)filter_mem + 15) & ~(uintptr_t)0x0F);
			void * img_mem = malloc(12 * node_size*node_size * sizeof(float) + 16);
			float *img_ptr = (float *)(((uintptr_t)img_mem + 16) & ~(uintptr_t)0x0F);
			void *imgout_mem = malloc(node_size*node_size * sizeof(float) + 16);
			float *imgout_ptr = (float *)(((uintptr_t)imgout_mem + 16) & ~(uintptr_t)0x0F);

			for (int k = 0; k < top_chans; k++) // input channels --- same as kernels_per_map - kern for each input
			{
				//unwrap_aligned(img_ptr, &top.node.x[k*kstep], jstep, 3);
				unwrap_aligned_3x3(img_ptr, &top.node.x[k*kstep], jstep);
				for (int map = 0; map < map_cnt; map++) // how many maps  maps= node.chans
				{
					memcpy(filter_ptr, &w.x[(map + k*maps)*kernel_size], 9 * sizeof(float));

					dot_unwrapped_3x3_sse(img_ptr, filter_ptr, imgout_ptr, outsize);

					float *out = node.x + map_size*map;
					for (int j = 0; j < outsize; j++) out[j] += imgout_ptr[j];
				}
			}
			free(filter_mem);
			free(img_mem);
			free(imgout_mem);
			return;
#endif //MOJO_SSE3
		}
		else
		{
			for(int map=0; map<maps; map++) // how many maps  maps= node.chans
			{
				for(int k=0; k<top_chans; k++) // input channels --- same as kernels_per_map - kern for each input
				{

					for(int j=0; j<node_size; j+= stride) // input h 
 						for(int i=0; i<node_size; i+= stride) // intput w
				
							node.x[i+(j)*node.cols +map_size*map]+= 
								unwrap_2d_dot(
									&top.node.x[(i)+(j)*jstep + k*kstep],
									&w.x[(map+k*maps)*kernel_size],
									kernel_cols,
									jstep,kernel_cols);
					
				}

			} //k
		} // all maps=chans
			
	}


#ifndef NO_TRAINING_CODE

	// convolution::distribute_delta
	virtual void distribute_delta(base_layer &top, const matrix &w, const int train=1)
	{
		
		// here to calculate top_delta += bottom_delta * W
//		top_delta.x[s] += bottom_delta.x[t]*w.x[s+t*w.cols];
		matrix delta_pad(delta, pad_cols, pad_rows);

		const int kstep=top.delta.cols*top.delta.rows;
		const int jstep=top.delta.cols;
		const int output_index=0;
		const int kernel_size=kernel_cols*kernel_rows;
		const int kernel_map_step = kernel_size*kernels_per_map;
		const int map_size=delta_pad.cols*delta_pad.rows;
		const float *_w = w.x;
		const int w_size = kernel_cols;
		const int delta_size = delta_pad.cols;
		const int map_cnt=maps;
		const int top_delta_size = top.delta.rows;
		const int top_delta_chans = top.delta.chans;
			
		if(kernel_cols==5)
		{					
#ifndef MOJO_SSE3
			for(int k=0; k<top_delta_chans; k++) // input channels --- same as kernels_per_map - kern for each input
			{ 
				_w=& w.x[k*maps*kernel_size];
				//continue;
				for(int map=0; map<map_cnt; map++) // how many maps  maps= node.chans
				{
					const float *_delta;
					_delta = &delta_pad.x[map*map_size];
					//_delta = &delta_pad.x[map*map_size];
					for(int j=0; j<top_delta_size; j+=1)// was stride) // input h 
					{
						for(int i=0; i<top_delta_size; i+=1)//stride) // intput w
						{
							const int td_i = i+(j)*jstep + k*kstep;
							top.delta.x[td_i] += unwrap_2d_dot_rot180_5x5( _delta+i, _w, delta_size,w_size);
						} // all input chans
						//output_index++;	
					_delta+=delta_size;
					} 
					_w+=kernel_size;
				}
			}
#else// MOJO_SSE3
			void * filter_mem = malloc(28 * sizeof(float) + 15);
			float *filter_ptr = (float *)(((uintptr_t)filter_mem + 15) & ~(uintptr_t)0x0F);
			void * img_mem = malloc(28 * delta_size*delta_size * sizeof(float) + 15);
			float *img_ptr = (float *)(((uintptr_t)img_mem + 15) & ~(uintptr_t)0x0F);
			void *imgout_mem = malloc(delta_size*delta_size * sizeof(float) + 15);
			float *imgout_ptr = (float *)(((uintptr_t)imgout_mem + 15) & ~(uintptr_t)0x0F);

			for (int map = 0; map<map_cnt; map++) // how many maps  maps= node.chans
			{
				unwrap_aligned(img_ptr, &delta_pad.x[map*map_size], delta_size,5);

				const int outsize = top_delta_size*top_delta_size;
				for (int k = 0; k<top_delta_chans; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_w = &w.x[(k*maps + map)*kernel_size];
					// flip, flip to make 180 version
					for (int ii = 0; ii < 25; ii++) filter_ptr[ii] = _w[24 - ii];

					dot_unwrapped_5x5_sse(img_ptr, filter_ptr, imgout_ptr, outsize);

					float *out = &top.delta.x[k*kstep];
					for (int j = 0; j < outsize; j++) out[j] += imgout_ptr[j];

				} // for map
			}
			free(imgout_mem);
			free(img_mem);
			free(filter_mem);

#endif // #ifndef MOJO_SSE3

		}
		else if(kernel_cols==3)					
		{
#ifndef MOJO_SSE3

			for(int k=0; k<top_delta_chans; k++) // input channels --- same as kernels_per_map - kern for each input
			{
				_w=& w.x[k*maps*kernel_size];
				//continue;
				for(int map=0; map<map_cnt; map++) // how many maps  maps= node.chans
				{
					const float *_delta;
					_delta = &delta_pad.x[map*map_size];
					//_delta = &delta_pad.x[map*map_size];
					for(int j=0; j<top_delta_size; j+=1)// was stride) // input h 
					{
						for(int i=0; i<top_delta_size; i+=1)//stride) // intput w
						{
							const int td_i = i+(j)*jstep + k*kstep;
							top.delta.x[td_i] += unwrap_2d_dot_rot180_3x3( _delta+i, _w, delta_size,w_size);

						} // all input chans
						//output_index++;	
					_delta+=delta_size;
					} 
					_w+=kernel_size;
				}
			}
#else// MOJO_SSE3
			void * filter_mem = malloc(12 * sizeof(float) + 15);
			float *filter_ptr = (float *)(((uintptr_t)filter_mem + 15) & ~(uintptr_t)0x0F);
			void * img_mem = malloc(12 * delta_size*delta_size * sizeof(float) + 15);
			float *img_ptr = (float *)(((uintptr_t)img_mem + 15) & ~(uintptr_t)0x0F);
			void *imgout_mem = malloc(delta_size*delta_size * sizeof(float) + 15);
			float *imgout_ptr = (float *)(((uintptr_t)imgout_mem + 15) & ~(uintptr_t)0x0F);

			for (int map = 0; map<map_cnt; map++) // how many maps  maps= node.chans
			{
				unwrap_aligned(img_ptr, &delta_pad.x[map*map_size], delta_size,3);

				const int outsize = top_delta_size*top_delta_size;
				for (int k = 0; k<top_delta_chans; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_w = &w.x[(k*maps + map)*kernel_size];
					for (int ii = 0; ii < 9; ii++) filter_ptr[ii] = _w[12 - ii];

					dot_unwrapped_3x3_sse(img_ptr, filter_ptr, imgout_ptr, outsize);

					float *out = &top.delta.x[k*kstep];
					for (int j = 0; j < outsize; j++) out[j] += imgout_ptr[j];

				} // for map
			}
			free(imgout_mem);
			free(img_mem);
			free(filter_mem);

#endif // #ifndef MOJO_SSE3
		}
		else
		{

			for(int j=0; j<top.delta.rows; j+=1) // input h 
			{
				for(int i=0; i<top.delta.cols; i+=1) // intput w
				{
					for(int k=0; k<top.delta.chans; k++) // input channels --- same as kernels_per_map - kern for each input
					{
						int td_i = i+(j)*jstep + k*kstep;
						for(int map=0; map<maps; map++) // how many maps  maps= node.chans
						{
							top.delta.x[td_i] += unwrap_2d_dot_rot180(
								&delta_pad.x[i+(j)*delta_pad.cols + map*map_size], 
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
		int kstep=top.delta.cols*top.delta.rows;
		int jstep=top.delta.cols;
		int output_index=0;
		int kernel_size=kernel_cols*kernel_rows;
		int kernel_map_step = kernel_size*kernels_per_map;
		int map_size=delta.cols*delta.rows;

		dw.resize(kernel_cols, kernel_rows,kernels_per_map*maps);
		dw.fill(0);
		
		// node x already init to 0
		output_index=0;
		const int top_node_size= top.node.cols;
		const int node_size = node.rows;
		const int delta_size = delta.cols;
		const int kern_len=kernel_cols;
		const float *_top;
		if(kern_len==5)
		{
			for(int map=0; map<maps; map++) // how many maps  maps= node.chans
			{
				const float *_delta =&delta.x[map*map_size];
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
				const float *_delta =&delta.x[map*map_size];
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
				const float *_delta =&delta.x[map*map_size];
				for(int k=0; k<top.node.chans; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_top = &top.node.x[k*kstep];
					const int w_i = (map+k*maps)*kernel_size;
					for(int jj=0; jj<kern_len; jj+=1)
					{
						for(int ii=0; ii<kern_len; ii+=1)
						{
							dw.x[w_i+ii+(jj)*kern_len]+= unwrap_2d_dot( _top + ii+(jj)*jstep, _delta,	node_size,top_node_size, delta_size);

						} // all input chans
					} // x
				} //y
			} // all maps=chans 
		}
	}

#endif
};


//----------------------------------------------------------------------------------------------------------
// C O N C A T I N A T I O N   
//
class concatination_layer : public base_layer
{
public:
	concatination_layer(const char *layer_name, int _w, int _h ) : base_layer(layer_name, _w,_h,1) { }
	virtual  ~concatination_layer(){}
	virtual void distribute_delta(base_layer &top, const matrix &w, const int train =1) {}
	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train =1) {}
	virtual void accumulate_signal(const base_layer &top_node, const matrix &w, const int train =0) {}

	virtual std::string get_config_string() {std::string str="concatination_layer "+int2str(node.cols)+" "+int2str(node.cols)+" "+int2str(node.cols)+"\n"; return str;}
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
	else if(str.compare("max_pool")==0)
	{
		iss >> c;  iss >> s;
		if(s>0 && s<=c)
			return new max_pooling_layer(layer_name, c, s);
		else
			return new max_pooling_layer(layer_name, c);
	}
	else if (str.compare("maxout_map_pool") == 0)
	{
		iss >> c;
		return new maxout_layer(layer_name, c);
	}

	else if (str.compare("semi_stochastic_pool") == 0)
	{
		iss >> c;  iss >> s;
		if (s>0 && s <= c)
			return new semi_stochastic_pooling_layer(layer_name, c, s);
		else
			return new semi_stochastic_pooling_layer(layer_name, c);
	}

	else if(str.compare("fractional_max_pool")==0)
	{
		iss>>c; 
		return new fractional_max_pooling_layer(layer_name, c);
	}
	else if(str.compare("convolution")==0)
	{
		std::string act;
		iss>>w;iss>>h;iss>>c; iss>>act; 
		return new convolution_layer(layer_name, w,h,c, new_activation_function(act));
	}
	else if (str.compare("dropout") == 0)
	{
		float fc;
		std::string act;
		iss >> fc;
		return new dropout_layer(layer_name, fc);
	}
	else if(str.compare("concatination")==0)
	{
		iss>>w;iss>>h;iss>>c;  
		return new concatination_layer(layer_name, w,h);
	}
	

	return NULL;
}


} // namespace
