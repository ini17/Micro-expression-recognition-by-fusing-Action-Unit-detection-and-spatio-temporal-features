function this = subsasgn(this, selector, value)
%SUBSASGN   Modify parameters of a robust function.
%   R = SUBSASGN(THIS, SEL, VAL) sets parameter of a robust function THIS
%   using given selector SEL to value VAL and returns modified object. 
%   Available selectors:
%      - ".type":  Function handle reflecting robust function type
%      - ".sigma": Sigma value (if appropriate, alias for param)
%      - ".param": Parameter
%  
%   This is a member function of the class 'robust_function'. 
%
%   Author:  Stefan Roth, Department of Computer Science, Brown University
%   Contact: roth@cs.brown.edu
%   $Date: $
%   $Revision: $

% Copyright 2004-2006 Brown University, Providence, RI.
% 
%                         All Rights Reserved
% 
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose other than its incorporation into a
% commercial product is hereby granted without fee, provided that the
% above copyright notice appear in all copies and that both that
% copyright notice and this permission notice appear in supporting
% documentation, and that the name of Brown University not be used in
% advertising or publicity pertaining to distribution of the software
% without specific, written prior permission.
% 
% BROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
% INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
% PARTICULAR PURPOSE.  IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR
% ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

  
  switch selector(1).type
   case '.'
    switch selector(1).subs
     case {'param', 'sigma'}
      this.param = value;

     otherwise
      error('This type of subscript is not supported.');
    end

   otherwise
    error('This type of subscript is not supported.');
  end